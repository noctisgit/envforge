"""Best-effort PII redaction for uploaded support conversations.

This runs BEFORE any conversation text is sent to an LLM or shown in the builder.
Each detected value is replaced by a typed placeholder that is consistent within a
conversation: the same email address becomes `<EMAIL_1>` every time it appears, a
second address becomes `<EMAIL_2>`, and so on. That keeps conversations readable
("I emailed <EMAIL_1> twice") without the real values.

What it catches (regex + validation):

    EMAIL     addresses
    PHONE     international formats (+44 20 7946 0958, (555) 123-4567, 0049 30 1234567,
              555-123-4567) and bare 7-15 digit numbers right after a phone cue
              ("call me on 5551234567"). Dates and plain IDs are left alone.
    CARD      13-19 digit credit/debit card numbers that pass the Luhn check
    IBAN      IBANs that pass the mod-97 check
    SSN       US social security numbers (123-45-6789, or 9 digits after "SSN")
    IP        public IPv4/IPv6 addresses (private, loopback and link-local addresses
              are network configuration, not personal data, and are kept unless
              `redact_private_ips=True`)
    MAC       MAC addresses (aa:bb:cc:dd:ee:ff, aa-bb-..., aabb.ccdd.eeff)
    URL       URLs carrying secrets (token=, key=, password=, sig=, long random path
              segments); ordinary URLs are kept
    ADDRESS   street-address-like text ("12 Baker Street, Apt 4"), text after
              "my address is" / "ship to" when it contains a number, UK postcodes
    DOB       dates after a label ("DOB:", "date of birth", "born on")
    NAME      names after cues ("my name is", "this is", "I'm", "Hi <Name>,",
              "Thanks, <Name>"), names in speaker labels ("Jane", "Agent Mike"), and
              every later mention of a name found that way in the same conversation
    ID        account/order/ticket numbers — only when `pseudonymize_ids=True`. They
              are not PII by default: the environment needs them as record keys.

What it does NOT catch (redaction is best-effort; review samples before sharing):

    - names without a cue ("Sarah said it was fine"), lower-case names after weak cues,
      nicknames, usernames inside free text, names of third parties
    - free-form addresses without a house number or street suffix, cities on their own,
      US ZIP codes on their own
    - phone numbers written without separators and without a cue, or spelled out
    - dates of birth without a label, ages, physical descriptions, health details
    - government IDs other than US SSNs (passport, NI numbers, ...) — add them with
      `extra_patterns`
    - anything in images, attachments or encodings (base64, URL-encoded text)
    - false positives: version strings that look like public IPs ("1.2.3.4"), long
      numeric IDs that happen to pass Luhn, capitalised words after "I'm"

Original values are never stored. Placeholder consistency uses salted SHA-256 hashes;
the salt is random per conversation unless you pass one, so hashes cannot be joined
across conversations or runs. Names found in a conversation are held in memory only
while that conversation is being redacted.
"""

from __future__ import annotations

import hashlib
import ipaddress
import re
import secrets
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field, replace

from envforge.builder.ingest import Conversation, Turn, normalize_role

PLACEHOLDER_RE = re.compile(r"(<[A-Z][A-Z0-9_]*_\d+>)")

ExtraPatterns = dict[str, "str | re.Pattern[str]"]


@dataclass
class RedactionResult:
    """Redacted text plus what was replaced.

    `counts_by_type` counts replacements (occurrences) per type, e.g. {"EMAIL": 2}.
    `mapping_hashes` maps each placeholder to a salted hash prefix of the value it
    replaced — enough to check consistency, useless for recovering the value.
    """

    text: str
    counts_by_type: dict[str, int] = field(default_factory=dict)
    mapping_hashes: dict[str, str] = field(default_factory=dict)


# ---- validators ----------------------------------------------------------------------------


def luhn_ok(digits: str) -> bool:
    """True if a digit string passes the Luhn checksum (used by payment cards)."""
    if not digits.isdigit():
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def iban_ok(value: str) -> bool:
    """True if a string is a structurally valid IBAN (ISO 13616 mod-97 check)."""
    s = re.sub(r"\s+", "", value).upper()
    if not (15 <= len(s) <= 34) or not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]+", s):
        return False
    rearranged = s[4:] + s[:4]
    numeric = "".join(str(int(ch, 36)) for ch in rearranged)
    return int(numeric) % 97 == 1


def _ip_is_public(value: str) -> bool | None:
    """True for a public IP, False for private/loopback/etc., None if not an IP."""
    try:
        ip = ipaddress.ip_address(value)
    except ValueError:
        return None
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_unspecified
        or ip.is_reserved
    )


# ---- patterns ------------------------------------------------------------------------------

_SECRET_KEYS = (
    "token|access_token|refresh_token|id_token|key|api_key|apikey|secret|password|passwd|pwd|"
    "pass|auth|authorization|sig|signature|session|sessionid|sid|code|otp|reset|jwt|hash"
)
URL_RE = re.compile(r"\b(?:https?://|www\.)[^\s<>\"']*[^\s<>\"'.,;:!?)\]]")
_URL_SECRET_QUERY = re.compile(rf"(?i)[?&#](?:{_SECRET_KEYS})=")
_URL_RANDOM_SEGMENT = re.compile(r"[/=]([A-Za-z0-9_\-]{20,})")

EMAIL_RE = re.compile(r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}(?:[ ]?[A-Z0-9]{1,3})?\b")
CARD_RE = re.compile(r"(?<![\w-])(?:\d[ -]?){12,18}\d(?![\w-])")
SSN_RE = re.compile(r"(?<![\d-])(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}(?![\d-])")
SSN_CUE_RE = re.compile(
    r"(?i)(?P<cue>\b(?:ssn|social security(?: number)?|soc\.? sec\.?)\b\D{0,15}?)(?P<value>\d{9})\b"
)
MAC_RE = re.compile(
    r"(?<![\w:-])[0-9A-Fa-f]{2}([:-])[0-9A-Fa-f]{2}(?:\1[0-9A-Fa-f]{2}){4}(?![\w:-])"
    r"|\b[0-9A-Fa-f]{4}\.[0-9A-Fa-f]{4}\.[0-9A-Fa-f]{4}\b"
)
IPV6_RE = re.compile(r"(?<![\w:])(?=[0-9A-Fa-f:]*:[0-9A-Fa-f:]*:)[0-9A-Fa-f:]{2,39}(?![\w:])")
IPV4_RE = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?!\d|\.\d)")

_MONTH = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
    r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)
_DATE = (
    r"(?:\d{1,4}[/.-]\d{1,2}[/.-]\d{1,4}"
    rf"|\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?{_MONTH}\.?,?\s+\d{{4}}"
    rf"|{_MONTH}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}})"
)
DOB_RE = re.compile(
    r"(?i)(?P<cue>\b(?:dob|d\.o\.b\.?|date of birth|birth\s?date|birthday|born on|born)"
    rf"\b\s*(?:is|was|:|-)?\s*)(?P<value>{_DATE})"
)

PHONE_RE = re.compile(r"(?<![\w/+.-])(?P<value>\+?\(?\d[\d \t().-]{5,22}\d)(?![\w])")
_PHONE_CUE = re.compile(
    r"(?i)\b(?:phone|tel|telephone|mobile|cell|call|text|sms|whatsapp|fax|reach me|"
    r"contact number|mobile number|cell number)\b[^.\n]{0,25}$"
)
_DATE_LIKE = re.compile(r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}")

_STREET = (
    r"(?:street|st|avenue|ave|road|rd|boulevard|blvd|lane|ln|drive|dr|court|ct|way|place|"
    r"pl|terrace|ter|close|crescent|cres|parkway|pkwy|highway|hwy|square|sq|circle|cir|"
    r"trail|row|mews|gardens|grove)"
)
ADDRESS_RE = re.compile(
    r"\b\d{1,5}[A-Za-z]?\s+(?:[A-Z0-9][\w'.-]*\s+){1,4}"
    rf"(?i:{_STREET})\b\.?"
    r"(?:,?\s*(?i:apt|apartment|suite|ste|unit|flat|#)\.?\s*[\w-]+)?"
    r"(?:,\s*[A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)?"
    r"(?:,?\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?)?"
)
ADDRESS_CUE_RE = re.compile(
    r"(?i)(?P<cue>\b(?:(?:my|the|our) (?:home |billing |shipping |delivery |mailing |postal |"
    r"new |old |service )?address (?:is|was|:)|i live at|i(?:'m| am) located at)\s*:?\s+)"
    r"(?P<value>[^.\n!?]{6,120})"
)
# "ship it to / delivered to …" is only an address when what follows looks like a street
# address — "delivered to the 4th floor mailroom" is not personal data.
SHIP_CUE_RE = re.compile(
    r"(?i)(?P<cue>\b(?:ship(?:ped)? (?:it )?to|deliver(?:ed)? (?:it )?to)\s*:?\s+)"
    rf"(?P<value>[^.\n!?]{{0,30}}?\d{{1,5}}[A-Za-z]?\s+(?:[A-Za-z0-9][\w'.-]*\s+){{0,4}}(?i:{_STREET})\b[^.\n!?]{{0,60}})"
)
UK_POSTCODE_RE = re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b")

_NAME_WORD = r"[A-Z][a-z]+(?:[-'’][A-Z]?[a-z]+)?"
_NAME = rf"{_NAME_WORD}(?: {_NAME_WORD}){{0,2}}"
NAME_STRONG_CUE_RE = re.compile(
    r"(?i:(?P<cue>\b(?:my name is|my name's|name\s*:|you(?:'re| are) (?:speaking|chatting|talking) "
    r"(?:to|with))\s+))(?P<value>[A-Za-z][a-z'’-]+(?: [A-Za-z][a-z'’-]+){0,2})"
)
NAME_CUE_RE = re.compile(rf"(?i:(?P<cue>\b(?:this is|i'?m|i am|it'?s me,?)\s+))(?P<value>{_NAME})")
NAME_GREETING_RE = re.compile(
    rf"(?im:(?P<cue>(?:^|[.!?]\s+)\s*(?:hi|hello|hey|dear|good (?:morning|afternoon|evening))"
    rf",?\s+))(?P<value>{_NAME})(?=\s*[,!.:\n]|$)"
)
# "— Tom Becker" / "-- Jane" at the end of a line (email signatures)
NAME_DASH_SIGNOFF_RE = re.compile(
    rf"(?m:(?P<cue>(?:^|\s)[—–-]{{1,2}}\s*))(?P<value>{_NAME})(?=\s*$)"
)
# "Ana Souza here, …" — a self-introduction at the start of a sentence
NAME_HERE_RE = re.compile(rf"(?m:(?P<cue>(?:^|[.!?,;]\s+)))(?P<value>{_NAME})(?=\s+here\b)")
NAME_SIGNOFF_RE = re.compile(
    rf"(?i:(?P<cue>\b(?:thanks|thank you|regards|best regards|kind regards|cheers|sincerely|"
    rf"many thanks|best wishes|best)\s*,?\s*))(?P<value>{_NAME})(?=\s*[.!,]?\s*$|\s*\n)"
)
_NAME_STOP = {
    # role words, greetings, pronouns and common capitalised non-names
    "customer", "agent", "support", "team", "sir", "madam", "ma'am", "all", "everyone",
    "there", "here", "the", "a", "an", "so", "really", "very", "just", "still", "not",
    "sorry", "glad", "happy", "back", "calling", "having", "trying", "using", "on", "in",
    "at", "from", "with", "your", "you", "me", "again", "fine", "good", "great", "ok",
    "okay", "yes", "no", "thanks", "thank", "hi", "hello", "hey", "afraid", "unable",
    "able", "sure", "going", "done", "now", "also", "and", "but", "or", "getting",
    "looking", "writing", "contacting", "reaching", "new", "interested", "worried",
    "concerned", "confused", "frustrated", "annoyed", "waiting", "currently", "always",
    "never", "already", "unfortunately", "hoping", "following", "wondering",
    "experiencing", "seeing", "told", "moving", "travelling", "traveling", "home",
    "away", "out", "over", "internet", "wifi", "router", "modem", "account", "order",
    "billing", "tech", "technical", "bot", "assistant", "guys", "folks", "service",
    "sales", "help", "desk", "helpdesk", "it", "this", "that", "what", "is", "my",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december", "today", "tomorrow", "yesterday",
    "morning", "afternoon", "evening", "much", "lot", "lots", "regards", "best",
    "cheers", "if", "when", "please", "pls", "operator", "rep", "staff", "user",
    "client", "caller", "admin", "system", "unknown", "speaking", "chatting", "being", "ready", "free", "available", "online", "offline",
}  # fmt: skip

_ID_CUE_RE = re.compile(
    r"(?i:(?P<cue>\b(?:order|account|acct|ticket|case|invoice|reference|ref|customer|member|"
    r"subscription|contract|booking|confirmation|tracking|serial|policy)"
    r"(?:\s*(?:number|no\.?|num|id|#))?\s*(?:is|:|#)?\s*))(?P<value>#?[A-Za-z0-9][A-Za-z0-9-]{3,})"
)
_ID_STANDALONE_RE = re.compile(r"\b[A-Z]{2,5}-?\d{4,}\b")


# ---- the session ---------------------------------------------------------------------------


def _sub_outside(pattern: re.Pattern[str], text: str, repl: Callable[[re.Match[str]], str]) -> str:
    """`pattern.sub(repl, text)`, but never inside existing placeholders."""
    parts = PLACEHOLDER_RE.split(text)
    for i in range(0, len(parts), 2):  # even indices are the text between placeholders
        if parts[i]:
            parts[i] = pattern.sub(repl, parts[i])
    return "".join(parts)


def _clean_name(value: str) -> str | None:
    """Trim a captured name at the first stopword; None if nothing name-like is left."""
    words: list[str] = []
    for w in value.split():
        if w.lower().strip("'’") in _NAME_STOP:
            break
        words.append(w)
    name = " ".join(words).strip(" ,.'’-")
    return name if len(name) >= 2 else None


class RedactionSession:
    """Placeholder state for one conversation.

    Use one session per conversation so placeholders restart at 1 and the same value
    always maps to the same placeholder within it.
    """

    def __init__(
        self,
        *,
        salt: bytes | None = None,
        pseudonymize_ids: bool = False,
        redact_private_ips: bool = False,
        extra_patterns: ExtraPatterns | None = None,
    ) -> None:
        self.salt = salt if salt is not None else secrets.token_bytes(16)
        self.pseudonymize_ids = pseudonymize_ids
        self.redact_private_ips = redact_private_ips
        self.extra_patterns: list[tuple[str, re.Pattern[str]]] = [
            (re.sub(r"[^A-Z0-9]+", "_", kind.upper()).strip("_") or "CUSTOM", re.compile(p))
            for kind, p in (extra_patterns or {}).items()
        ]
        self._by_hash: dict[str, str] = {}  # salted hash -> placeholder
        self._next: Counter[str] = Counter()
        self.counts: Counter[str] = Counter()
        self.hashes: dict[str, str] = {}  # placeholder -> salted hash prefix
        # names found in this conversation, held in memory only while redacting
        self._names: dict[str, str] = {}  # lower-case name -> placeholder
        self._names_re: re.Pattern[str] | None = None

    # -- placeholders --
    def _digest(self, kind: str, value: str) -> str:
        norm = value.strip().lower()
        if kind in ("PHONE", "CARD", "SSN"):
            norm = re.sub(r"\D", "", norm)
        elif kind in ("IBAN", "MAC"):
            norm = re.sub(r"[^0-9a-z]", "", norm)
        return hashlib.sha256(self.salt + f"{kind}\x00{norm}".encode()).hexdigest()

    def placeholder(self, kind: str, value: str, *, count: bool = True) -> str:
        """The placeholder for a value, allocating `<KIND_n>` on first sight."""
        digest = self._digest(kind, value)
        ph = self._by_hash.get(digest)
        if ph is None:
            self._next[kind] += 1
            ph = f"<{kind}_{self._next[kind]}>"
            self._by_hash[digest] = ph
            self.hashes[ph] = digest[:16]
        if count:
            self.counts[kind] += 1
        return ph

    # -- names --
    def add_name(self, name: str) -> str | None:
        """Register a person's name (and its parts) so every mention is redacted.
        Returns its placeholder, or None if it does not look like a name."""
        cleaned = _clean_name(name)
        if not cleaned:
            return None
        key = cleaned.lower()
        if key in self._names:
            return self._names[key]
        # "Jane Doe" after "Jane": same person, same placeholder
        known_part = next((self._names[p] for p in key.split() if p in self._names), None)
        ph = known_part or self.placeholder("NAME", key, count=False)
        self._names[key] = ph
        for part in key.split():
            if len(part) >= 3 and part not in _NAME_STOP:
                self._names.setdefault(part, ph)
        self._names_re = None
        return ph

    def find_names(self, text: str) -> list[str]:
        """Names introduced by cues in `text` ("my name is Jane", "Thanks, Bob")."""
        found: list[str] = []
        for rx in (
            NAME_STRONG_CUE_RE,
            NAME_CUE_RE,
            NAME_GREETING_RE,
            NAME_SIGNOFF_RE,
            NAME_DASH_SIGNOFF_RE,
            NAME_HERE_RE,
        ):
            for m in rx.finditer(text):
                name = _clean_name(m.group("value"))
                if name:
                    found.append(name)
        return found

    def _replace_known_names(self, text: str) -> str:
        if not self._names:
            return text
        if self._names_re is None:
            alts = sorted(self._names, key=len, reverse=True)
            self._names_re = re.compile(
                r"(?i)(?<![\w<])(" + "|".join(re.escape(a) for a in alts) + r")(?![\w>])"
            )

        def repl(m: re.Match[str]) -> str:
            self.counts["NAME"] += 1
            return self._names[m.group(1).lower()]

        return _sub_outside(self._names_re, text, repl)

    # -- the pipeline --
    def _typed(self, kind: str, check: Callable[[str], bool] | None = None) -> Callable:
        """A re.sub callback replacing the `value` group (or the whole match)."""

        def repl(m: re.Match[str]) -> str:
            has_value = "value" in m.re.groupindex and m.group("value") is not None
            value = m.group("value") if has_value else m.group(0)
            if check is not None and not check(value):
                return m.group(0)
            ph = self.placeholder(kind, value)
            if has_value:
                start, end = m.span("value")
                base = m.start()
                whole = m.group(0)
                return whole[: start - base] + ph + whole[end - base :]
            return ph

        return repl

    def _url_check(self, url: str) -> bool:
        if _URL_SECRET_QUERY.search(url):
            return True
        for seg in _URL_RANDOM_SEGMENT.findall(url):
            if re.search(r"\d", seg) and re.search(r"[A-Za-z]", seg):
                return True
        return False

    def _ip_check(self, value: str) -> bool:
        public = _ip_is_public(value)
        if public is None:
            return False
        return public or self.redact_private_ips

    @staticmethod
    def _card_check(value: str) -> bool:
        digits = re.sub(r"\D", "", value)
        return 13 <= len(digits) <= 19 and digits[0] in "23456" and luhn_ok(digits)

    def _phone_repl(self, m: re.Match[str]) -> str:
        value = m.group("value").strip()
        digits = re.sub(r"\D", "", value)
        if not (7 <= len(digits) <= 15) or _DATE_LIKE.fullmatch(value):
            return m.group(0)
        groups = [g for g in re.split(r"[\s().-]+", value) if g]
        if "." in value and any(not 2 <= len(g) <= 4 for g in groups):
            return m.group(0)  # a version string like 10.0.19045.3803, not 555.123.4567
        international = value.startswith("+") or (value.startswith("00") and len(digits) >= 10)
        formatted = ("(" in value and ")" in value) or (len(groups) >= 3 and len(digits) >= 9)
        # a local number like 555-0173 (3 + 4 digits with a separator)
        formatted = formatted or bool(re.fullmatch(r"\d{3}[-. ]\d{4}", value))
        before = m.string[max(0, m.start() - 40) : m.start()]
        cued = bool(_PHONE_CUE.search(before))
        if international or formatted or cued:
            return m.group(0).replace(m.group("value"), self.placeholder("PHONE", value))
        return m.group(0)

    def redact(self, text: str) -> RedactionResult:
        """Redact one piece of text with this session's placeholders."""
        before = Counter(self.counts)
        for kind, rx in self.extra_patterns:
            text = _sub_outside(rx, text, self._typed(kind))
        text = _sub_outside(URL_RE, text, self._typed("URL", self._url_check))
        text = _sub_outside(EMAIL_RE, text, self._typed("EMAIL"))
        text = _sub_outside(IBAN_RE, text, self._typed("IBAN", iban_ok))
        text = _sub_outside(CARD_RE, text, self._typed("CARD", self._card_check))
        text = _sub_outside(SSN_RE, text, self._typed("SSN"))
        text = _sub_outside(SSN_CUE_RE, text, self._typed("SSN"))
        text = _sub_outside(MAC_RE, text, self._typed("MAC"))
        text = _sub_outside(IPV6_RE, text, self._typed("IP", self._ip_check))
        text = _sub_outside(IPV4_RE, text, self._typed("IP", self._ip_check))
        text = _sub_outside(DOB_RE, text, self._typed("DOB"))
        text = _sub_outside(PHONE_RE, text, self._phone_repl)
        text = _sub_outside(SHIP_CUE_RE, text, self._typed("ADDRESS"))
        text = _sub_outside(
            ADDRESS_CUE_RE, text, self._typed("ADDRESS", lambda v: bool(re.search(r"\d", v)))
        )
        text = _sub_outside(ADDRESS_RE, text, self._typed("ADDRESS"))
        text = _sub_outside(UK_POSTCODE_RE, text, self._typed("ADDRESS"))
        for name in self.find_names(text):
            self.add_name(name)
        text = self._replace_known_names(text)
        if self.pseudonymize_ids:
            text = _sub_outside(
                _ID_CUE_RE, text, self._typed("ID", lambda v: bool(re.search(r"\d", v)))
            )
            text = _sub_outside(_ID_STANDALONE_RE, text, self._typed("ID"))
        counts = {k: v - before.get(k, 0) for k, v in self.counts.items() if v - before.get(k, 0)}
        used = set(PLACEHOLDER_RE.findall(text))
        return RedactionResult(
            text=text,
            counts_by_type=counts,
            mapping_hashes={ph: h for ph, h in self.hashes.items() if ph in used},
        )

    def redact_speaker(self, label: str | None) -> str | None:
        """'Jane' -> '<NAME_1>', 'Agent Mike' -> 'Agent <NAME_2>', 'Customer' unchanged."""
        if not label:
            return label
        words = label.split()
        role_words = [w for w in words if normalize_role(w) != "unknown"]
        name_words = [w for w in words if normalize_role(w) == "unknown"]
        name = " ".join(name_words).strip(" ()-–:")
        if not name:
            return label
        ph = self.add_name(name) or self.placeholder("NAME", name, count=False)
        self.counts["NAME"] += 1
        return " ".join([*role_words, ph]) if role_words else ph


# ---- public helpers ------------------------------------------------------------------------


def redact_text(
    text: str,
    *,
    pseudonymize_ids: bool = False,
    redact_private_ips: bool = False,
    extra_patterns: ExtraPatterns | None = None,
    known_names: list[str] | None = None,
    salt: bytes | None = None,
) -> RedactionResult:
    """Redact a single piece of text (its own placeholder numbering).

    `extra_patterns` maps a type name to a regex, e.g.
    {"PASSPORT": r"\\b[A-Z]\\d{8}\\b"}; if the regex has a group named `value`, only
    that group is replaced. `known_names` are redacted wherever they appear.
    """
    session = RedactionSession(
        salt=salt,
        pseudonymize_ids=pseudonymize_ids,
        redact_private_ips=redact_private_ips,
        extra_patterns=extra_patterns,
    )
    for name in known_names or []:
        session.add_name(name)
    return session.redact(text)


def redact_conversation(
    conv: Conversation,
    *,
    pseudonymize_ids: bool = False,
    redact_private_ips: bool = False,
    extra_patterns: ExtraPatterns | None = None,
    salt: bytes | None = None,
) -> tuple[Conversation, dict[str, int]]:
    """Redact every turn (text and speaker label) with one consistent session.

    Two passes: first collect names from speaker labels and name cues across the whole
    conversation, then redact each turn — so "Hi Jane" in turn 1 is caught even if
    Jane only says "my name is Jane" in turn 5.

    Returns a new conversation (the input is not modified) with
    `meta["redacted"] = True` and `meta["redaction_counts"]`, plus the counts.
    """
    session = RedactionSession(
        salt=salt,
        pseudonymize_ids=pseudonymize_ids,
        redact_private_ips=redact_private_ips,
        extra_patterns=extra_patterns,
    )
    speakers = [session.redact_speaker(t.speaker) for t in conv.turns]
    for t in conv.turns:
        for name in session.find_names(t.text):
            session.add_name(name)
    turns = [
        Turn(role=t.role, text=session.redact(t.text).text, ts=t.ts, speaker=spk)
        for t, spk in zip(conv.turns, speakers, strict=True)
    ]
    counts = dict(session.counts)
    meta = {k: v for k, v in conv.meta.items() if k not in ("redaction_counts",)}
    meta.update(redacted=True, redaction_counts=counts)
    conv_id = conv.id
    if session.redact(conv.id).text != conv.id:  # an email or phone number used as the id
        conv_id = "conv-" + session._digest("ID", conv.id)[:12]  # salted: not reversible
        meta["id_redacted"] = True
    return replace(conv, id=conv_id, turns=turns, meta=meta), counts


def redact_conversations(
    conversations: list[Conversation], **options: object
) -> tuple[list[Conversation], dict[str, int]]:
    """Redact many conversations (one session each). Returns them and total counts."""
    out: list[Conversation] = []
    totals: Counter[str] = Counter()
    for conv in conversations:
        redacted, counts = redact_conversation(conv, **options)  # type: ignore[arg-type]
        out.append(redacted)
        totals.update(counts)
    return out, dict(totals)
