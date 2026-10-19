"""Best-effort PII/PHI redaction for uploaded support conversations and exports.

This runs BEFORE any conversation text is sent to an LLM or shown in the builder, and
again (via `sanitize_json_strings`) on anything that leaves the system: published
environments and exported training data. Each detected value is replaced by a typed
placeholder that is consistent within a conversation: the same email address becomes
`<EMAIL_1>` every time it appears, a second address becomes `<EMAIL_2>`, and so on.
That keeps conversations readable ("I emailed <EMAIL_1> twice") without the real values.

It is a deterministic, rule-based technical safeguard. It is NOT a HIPAA (or GDPR)
certification: a person must still review samples (see `build_redaction_report`)
before data is shared, and the "strict" profile only aims at the Safe Harbor list.

Profiles
--------

Every public function takes `profile="standard"` (the default), `"strict"` or `"off"`.

off       — no redaction at all, for public or synthetic datasets whose "customers" are
            fictional (e.g. ABCD). Text is returned unchanged, but conversations are
            still marked `meta["redacted"] = True` with `meta["redaction_profile"] =
            "off"`: the corpus store only accepts marked conversations, and "off" is an
            explicit choice a person made in the UI, recorded with the data.
standard  — tuned for tech-support logs: removes contact details, payment data and
            names, but keeps what an environment needs (dates, order numbers, private IPs,
            ordinary URLs, cities).
strict    — aims at the 18 identifier categories of HIPAA's Safe Harbor method
            (45 CFR 164.514(b)(2)). Everything standard does, plus the rules marked
            "strict" below. Strict is always a superset of standard.

What standard catches (regex + validation):

    EMAIL     addresses
    PHONE     international formats (+44 20 7946 0958, (555) 123-4567, 0049 30 1234567,
              555-123-4567) and bare 7-15 digit numbers right after a phone/fax cue
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

Strict profile and HIPAA Safe Harbor
------------------------------------

    Safe Harbor identifier          strict rule (placeholder)                      coverage
    ------------------------------  ---------------------------------------------  --------
    A  Names                        cue names, speaker labels, titles ("Dr Lee",   partial;
                                    "Patient Smith"), NER PERSON when ner=True     good w/ NER
       (NAME)
    B  Geography below state        street addresses + UK postcodes (ADDRESS),     partial
                                    "City, ST" / "City, State" and NER LOCATION
                                    (LOCATION), US ZIPs after a state or a "zip"
                                    cue -> first 3 digits ("021XX"; "000XX" for
                                    the 17 low-population prefixes HHS lists)
    C  Dates (except year), ages>89 numeric, ISO and month-name dates ->           good for
                                    "<DATE_n> 2024" (year kept); DOB -> <DOB_n>;   common
                                    a year 90+ years ago is dropped too; ages      formats
                                    over 89 -> "90+"; NER DATE_TIME when ner=True
    D  Telephone numbers            PHONE                                          good
    E  Fax numbers                  PHONE (fax is a phone cue)                     good
    F  Email addresses              EMAIL                                          good
    G  Social security numbers      SSN                                            good
    H  Medical record numbers       cue: MRN, medical record, chart, patient id    cue-based
       (MRN)                        + any token with 6+ digits (ID)
    I  Health plan beneficiary nos. cue: member/subscriber/beneficiary/policy/     cue-based
       (HEALTH_PLAN)                group/insurance/Medicare/Medicaid/MBI + ID rule
    J  Account numbers (ACCOUNT)    cue: account/acct/routing + CARD/IBAN + ID     cue-based
    K  Certificate/license numbers  cue: license/licence/DL/DEA/NPI/certificate/   cue-based
       (LICENSE)                    passport/permit + NER licence recognisers
    L  Vehicle ids, licence plates  17-character VINs; plates after a cue          partial
       (VEHICLE)                    ("plate", "registration", "reg")
    M  Device ids, serial numbers   cue: serial, S/N, IMEI, MEID, ESN, UDID,       cue-based
       (DEVICE)                     service tag; MAC addresses; ID rule
    N  URLs                         every URL (URL)                                good
    O  IP addresses                 every IP, private ones included (IP)           good
    P  Biometric identifiers        cannot be detected in text                     none
    Q  Full-face photographs        out of scope: images/attachments are not       none
                                    processed here
    R  Any other unique number,     tokens with 6+ digits (ID), `extra_patterns`;  partial
       characteristic or code       identifying characteristics ("the only
                                    one-armed pilot in town") cannot be detected

Safe Harbor also requires that the covered entity has no actual knowledge that the
remaining information could identify someone. No code can certify that.

keep_patterns: strict mode would also destroy the record keys an environment needs
("HD-1001", "PRN-302", "C-1101"), so values matching `keep_patterns` are never touched
by any rule. The default for strict is `DEFAULT_KEEP_PATTERNS`: an upper-case prefix
of 1-6 letters, a dash, and digits — except prefixes that are themselves health
identifiers (MRN-, SSN-, MBI-, NPI-, MD-, ...). Pass `keep_patterns=()` to keep nothing, or
your own list. The standard profile keeps nothing extra by default (it does not
redact such keys in the first place unless `pseudonymize_ids=True`).

Optional NER layer
------------------

With `ner=True` a local named-entity recogniser (Microsoft Presidio + spaCy; see
`envforge.builder.ner`) also finds names, places and (strict) dates without cues. Its
findings are fed into the same placeholder machinery: a name NER found and the same
name found after "my name is" share one `<NAME_n>`. If Presidio or the spaCy model is
missing, redaction silently proceeds regex-only and `meta["redaction_ner"]` is False.

What it does NOT catch (redaction is best-effort; review samples before sharing):

    - without NER: names without a cue ("Sarah said it was fine"), cities on their
      own; with NER: lower-case or unusual names the model misses
    - lower-case names after weak cues, nicknames, usernames inside free text
    - free-form addresses without a house number or street suffix; US ZIP codes with
      no state or cue before them (strict: caught by nothing — a bare 5-digit number
      is indistinguishable from a quantity)
    - phone numbers written without separators and without a cue, or spelled out
      (strict catches them as ID when they have 6+ digits)
    - relative dates ("last Tuesday", "two weeks ago"), ages written in words
    - government IDs other than US SSNs in standard (add them with `extra_patterns`)
    - anything in images, attachments or encodings (base64, URL-encoded text)
    - false positives: version strings that look like public IPs ("1.2.3.4"), long
      numeric IDs that happen to pass Luhn, capitalised words after "I'm"; in strict,
      error codes and KB numbers with 6+ digits become <ID_n> (add a keep pattern)

Original values are never stored. Placeholder consistency uses salted SHA-256 hashes;
the salt is random per conversation unless you pass one, so hashes cannot be joined
across conversations or runs. Names found in a conversation are held in memory only
while that conversation is being redacted.

Surrogates instead of placeholders
----------------------------------

With `surrogates=True` each detected value is replaced by a consistent, realistic FAKE
value instead of `<KIND_n>`: "Crystal Minh" -> "Imani Okafor", "cminh730" -> "iokafor418",
"cminh730@email.com" -> "iokafor418@example.org", "(977) 625-2661" -> "(415) 555-0137".
Why: training data full of `<NAME_1>` teaches a model to write placeholders and never
shows it a real-looking name, email or phone number to handle. Natural text keeps the
conversation's rhythm and lets the model learn to *use* identifiers ("Thanks, Imani —
I've found the account for iokafor418") without learning anyone's actual identifiers.
Every surrogate is fake by construction:

    NAME      from built-in, culturally diverse first/last-name lists; one surrogate person
              per real person, so a first-name-only mention ("Thanks, Crystal") becomes
              that person's surrogate first name; the original's case is kept
    USERNAME  the conversation's surrogate name + fresh digits ("iokafor418")
    EMAIL     a surrogate handle @ example.com / example.org / example.net (RFC 2606)
    PHONE     the fictional 555-01xx range, in the original's format
    ADDRESS   a fictional street + a generic town (Springfield, Fairview, …) + state + ZIP;
              strict keeps its ZIP3 rule ("123XX")
    LOCATION  a generic town name
    CARD      a Luhn-valid test number (4111 …), same length and grouping
    IP        documentation ranges: 192.0.2.x / 198.51.100.x / 203.0.113.x, 2001:db8::
    MAC, IDs  same shape: letters -> letters, digits -> digits, separators kept
    DATE, DOB placeholders as before (a made-up date could change a timeline's meaning:
              "bought in November, so outside the 90 days"); NRP keeps its placeholder too

Surrogates are deterministic: seeded from the session salt and the value's salted
digest, so the same value always maps to the same surrogate within a session (and, with
the default random salt, to an unrelated one in the next conversation). A surrogate is
never equal to the value it replaces, and two different values of one type never share
a surrogate. Counts are still reported by type. Internally the pipeline still works with
placeholders (so every protection rule applies unchanged) and renders surrogates last.

Tier 2: a local span model, with risk triage
--------------------------------------------

`redact_conversations(..., tier2="auto")` adds a second, much slower detector
(`envforge.builder.pii_model`: GLiNER on CPU, ~100 ms per turn) only where tier 1 is
unsure. Tier 1 runs on every conversation; `risk(before, after)` scores what it left
behind (an unredacted answer to "your name?", capitalised word pairs, letter+digit
tokens, "@", long digit runs, NER being off, …); conversations scoring at least
`tier2_threshold` go through the model in one batch, its spans are filtered for false
positives (`pii_model.filter_spans`) and fed into the same placeholder / surrogate
machinery as NER (a name found by both shares one placeholder; nothing is replaced
twice). `meta["redaction_tier2"]` records whether a conversation went through tier 2,
`meta["redaction_risk"]` / `meta["redaction_risk_reasons"]` why. `tier2="all"` sends
everything; if the model is unavailable, tier 1's result is kept and nothing crashes.

Auditing
--------

`scan_text` is a detector-only pass that lists what still *looks like* PII in text that
has already been redacted. `build_redaction_report` / `audit` combine counts, NER
status, a sample of redacted snippets and those suspicious remnants into a dict for a
human sign-off step. `sanitize_json_strings` re-redacts every string in a JSON-like
structure (task records, chat-format training examples) before export.
"""

from __future__ import annotations

import hashlib
import ipaddress
import random
import re
import secrets
import time
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import date
from typing import Any, Literal

from envforge.builder import ner as ner_layer
from envforge.builder import pii_model
from envforge.builder.ingest import Conversation, Turn, normalize_role

Profile = Literal["off", "standard", "strict"]
PROFILES: tuple[str, ...] = ("off", "standard", "strict")
Tier2Mode = Literal["off", "auto", "all"]
DEFAULT_TIER2_THRESHOLD = 1.0

PLACEHOLDER_RE = re.compile(r"(<[A-Z][A-Z0-9_]*_\d+>)")

ExtraPatterns = dict[str, "str | re.Pattern[str]"]
KeepPatterns = Sequence["str | re.Pattern[str]"]

# Record keys like "HD-1001", "PRN-302", "C-1101", "INC-0042": an upper-case prefix, a
# dash, digits. Prefixes that are health identifiers in their own right are excluded so
# "MRN-1234567" is still redacted. Not glued to an email ("HD-1001@corp.com").
DEFAULT_KEEP_PATTERNS: tuple[str, ...] = (
    r"(?<![\w@.-])(?!(?:MRN|MR|SSN|NPI|DEA|DL|MBI|HICN|MBR|MEM|PT|DOB|ACCT|PHN|MD|RN|LIC|CERT|VIN)-)"
    r"[A-Z]{1,6}-\d{1,9}(?![\w@-])",
)

# 3-digit ZIP prefixes whose area had 20,000 people or fewer (2000 Census), which Safe
# Harbor says must become 000 (HHS de-identification guidance, section 2.4).
RESTRICTED_ZIP3: frozenset[str] = frozenset(
    "036 059 063 102 203 556 692 790 821 823 830 831 878 879 884 890 893".split()
)

_THIS_YEAR = date.today().year


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


def _has_digit(value: str) -> bool:
    return any(ch.isdigit() for ch in value)


def _id_value_ok(value: str) -> bool:
    """A cue-introduced value is an identifier if it has a digit and is not just a
    year ("insurance 2024 renewal")."""
    return _has_digit(value) and not re.fullmatch(r"(?:19|20)\d\d", value)


# ---- patterns: standard --------------------------------------------------------------------

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
    # \s+ between cue words: chat exports often contain doubled spaces ("my  name  is")
    r"(?i:(?P<cue>\b(?:my\s+name\s+is|my\s+name's|name\s*:|you(?:'re|\s+are)\s+(?:speaking|chatting|talking)\s+"
    r"(?:to|with))\s+))(?P<value>[A-Za-z][a-z'’-]+(?:[ \t]{1,3}[A-Za-z][a-z'’-]+){0,2})"
)
NAME_CUE_RE = re.compile(
    rf"(?i:(?P<cue>\b(?:this\s+is|i'?m|i\s+am|it'?s\s+me,?)\s+))(?P<value>{_NAME})"
)
NAME_GREETING_RE = re.compile(
    rf"(?im:(?P<cue>(?:^|[.!?]\s+)\s*(?:hi|hello|hey|dear|good\s+(?:morning|afternoon|evening))"
    rf",?\s+))(?P<value>{_NAME})(?=\s*[,!.:\n]|$)"
)
# "— Tom Becker" / "-- Jane" at the end of a line (email signatures)
NAME_DASH_SIGNOFF_RE = re.compile(
    rf"(?m:(?P<cue>(?:^|\s)[—–-]{{1,2}}\s*))(?P<value>{_NAME})(?=\s*$)"
)
# "Ana Souza here, …" — a self-introduction at the start of a sentence
NAME_HERE_RE = re.compile(rf"(?m:(?P<cue>(?:^|[.!?,;]\s+)))(?P<value>{_NAME})(?=\s+here\b)")
NAME_SIGNOFF_RE = re.compile(
    rf"(?i:(?P<cue>\b(?:thanks|thank\s+you|regards|best\s+regards|kind\s+regards|cheers|sincerely|"
    rf"many\s+thanks|best\s+wishes|best)\s*,?\s*))(?P<value>{_NAME})(?=\s*[.!,]?\s*$|\s*\n)"
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


# ---- patterns: strict ----------------------------------------------------------------------

# Words that are often capitalised but are not names. Used to keep NER and title cues
# from registering "Will" or "Name" as a person (names are matched case-insensitively
# once registered), and by `scan_text` to tell name-like bigrams from ordinary ones.
COMMON_WORDS: frozenset[str] = frozenset(_NAME_STOP) | frozenset(
    """
    about above after again against ago ahead almost alone along am among any anyone
    anything are around as ask asked asking because been before behind below between
    both by can cannot could did do does doing down during each either else enough even
    ever every few first for further get gets got had has have he her hers him his how
    however i into its itself last later least less let like little long made make many
    maybe might mine more most must my myself near need needs neither next nor of off
    often once one only other others our ours ourselves own past per perhaps quite
    rather right same she should since some someone something sometimes soon such than
    their theirs them then these they thing things those though through thus till to
    too under until up upon us was we well were where whether which while who whom whose
    why will with within without would yet yours yourself
    add added address admit admitted advice advised alert allergy allergies appointment
    approved area asset assigned attached battery bill blood board book booked break
    browser bug building call called camera cancel cancelled card care case cause change
    changed charge charged chart check checked chrome claim clinic close closed cloud
    code connect connected connection contact copay cost could cough country credit
    data date day days dear delete department desktop device diagnosis discharge
    disk display doctor dose down download drive driver due email emergency end error
    escalate escalated event exam excel extension failed fax fee fever file files
    firmware fix fixed floor follow form found general group hardware headset health
    history hospital hub id insurance install installed invoice issue issues item key
    keyboard lab laptop level license line link list local locked login logon mail
    manager medical medication meeting member message microsoft mobile model monitor
    mouse name network note notes number nurse office open outage outlook page pain
    paid password patient payment pending phone plan please plus policy port portal
    power prescription printer priority pro problem profile program queue record records
    refund remote renew repair report request requested reset restart result results
    room rule scan screen server service session settings setup sign software start
    status step storage subscription symptom symptoms system table teams test text
    ticket time token treatment unit update updated upgrade urgent usb version visit
    vpn ward warranty web week weeks windows wireless work year years zoom
    apple android google iphone ipad mac linux office word powerpoint onedrive
    sharepoint slack adobe dell lenovo cisco hp
    north south east west street road avenue city state county
    mr mrs ms dr doctor prof
    """.split()
)

_STATE_ABBR = (
    "AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE "
    "NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY PR"
).split()
_STATE_NAMES = (
    "Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|Delaware|Florida|"
    "Georgia|Hawaii|Idaho|Illinois|Indiana|Iowa|Kansas|Kentucky|Louisiana|Maine|Maryland|"
    "Massachusetts|Michigan|Minnesota|Mississippi|Missouri|Montana|Nebraska|Nevada|"
    "New Hampshire|New Jersey|New Mexico|New York|North Carolina|North Dakota|Ohio|"
    "Oklahoma|Oregon|Pennsylvania|Rhode Island|South Carolina|South Dakota|Tennessee|"
    "Texas|Utah|Vermont|Virginia|Washington|West Virginia|Wisconsin|Wyoming"
)
# "ID 12345" is far more often an identifier than Idaho, so ID is left out here.
_ZIP_STATES = "|".join(s for s in _STATE_ABBR if s != "ID")
ZIP_STATE_RE = re.compile(rf"\b(?:{_ZIP_STATES})\.?\s+(?P<value>\d{{5}}(?:-\d{{4}})?)(?![\w-])")
ZIP_CUE_RE = re.compile(
    r"(?i:(?P<cue>\b(?:zip(?:\s*code)?|postal\s*code|post\s*code)\s*(?:is|:|#)?\s*))"
    r"(?P<value>\d{5}(?:-\d{4})?)(?![\w-])"
)
CITY_STATE_RE = re.compile(
    r"(?P<value>\b[A-Z][a-z]+(?:[ -][A-Z][a-z]+){0,2}),\s*"
    rf"(?=(?:{'|'.join(_STATE_ABBR)})\b(?!['’])|(?:{_STATE_NAMES})\b)"
)

# Dates in the formats people actually type. Each is validated (month 1-12, day 1-31)
# before it is replaced, so "10-20-30" is a date but "12-34-56" is not.
_ORD = r"(?:st|nd|rd|th)?"
_MONTH_CAP = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|"
    r"Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)
STRICT_DATE_RE = re.compile(
    r"(?<![\w/.:-])(?:"
    # 2024-05-01, 2024-05-01T10:00:00Z, 2024/05/01, 2024.05.01
    r"\d{4}-\d{1,2}-\d{1,2}(?:[T ]\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?"
    r"|\d{4}[/.]\d{1,2}[/.]\d{1,2}"
    # 2026-09 and 09/2026 (a month is a date element too)
    r"|\d{4}-\d{2}|\d{1,2}/\d{4}"
    # 05/01/2024, 5-1-24, 01.05.2024 (dots only with a 4-digit year: "1.2.24" is a version)
    r"|\d{1,2}[/-]\d{1,2}[/-](?:\d{4}|\d{2})|\d{1,2}\.\d{1,2}\.\d{4}"
    # 1 May 2024, 1st of may, 2024, 3 March (without a year the month must be
    # capitalised: "option 5 may work" is not a date)
    rf"|\d{{1,2}}{_ORD}\s+(?:of\s+)?(?:(?i:{_MONTH})\.?,?\s+\d{{4}}|{_MONTH_CAP}\b)"
    # May 1, 2024, may 1st 2024, Sept. 3
    rf"|(?i:{_MONTH})\.?\s+\d{{1,2}}{_ORD},?\s+\d{{4}}"
    rf"|{_MONTH_CAP}\.?\s+\d{{1,2}}{_ORD}(?![\d:])"
    # May 2024, March, 2024
    rf"|(?i:{_MONTH})\.?,?\s+\d{{4}}"
    r")(?![\w/-]|\.\d)"
)
# "born in 1931": a bare year that, for a birth, implies an age over 89.
BIRTH_YEAR_RE = re.compile(r"(?i:(?P<cue>\b(?:born|b\.)\s+(?:in\s+)?))(?P<value>1[89]\d\d)\b")
AGE_SUFFIX_RE = re.compile(
    r"(?i)\b(?P<value>9\d|1[0-4]\d)(?!\+)"
    r"(?=[\s-]*(?:years?|yrs?)[\s-]*old\b|[\s-]*year[\s-]*old\b|\s*(?:y/?o|y\.o\.)(?!\w))"
)
AGE_CUE_RE = re.compile(
    r"(?i:(?P<cue>\b(?:age[ds]?|aged)\s*(?:is|of|:|-)?\s*))(?P<value>9\d|1[0-4]\d)\b(?!\+)"
)
# Name words for the strict rules also cover "O'Neil" and "McDonald".
_NAME_WORD_X = r"(?:[A-Z]['’])?[A-Z][a-z]+(?:[A-Z][a-z]+)?(?:[-'’][A-Z]?[a-z]+)?"
_NAME_X = rf"{_NAME_WORD_X}(?: {_NAME_WORD_X}){{0,2}}"
NAME_TITLE_RE = re.compile(
    rf"(?P<cue>\b(?:Dr|Mr|Mrs|Ms|Miss|Mx|Prof|Nurse|(?i:patient|pt))[.:]?\s+)(?P<value>{_NAME_X})"
)
# "Thanks Tom, ..." / "ok hi Jane" anywhere in a sentence (standard needs the name at the
# end of the line or the greeting at the start of a sentence).
NAME_LOOSE_GREETING_RE = re.compile(
    rf"(?P<cue>\b(?i:thanks|thank you|thx|hi|hello|hey|dear|bye|cheers)\s*,?\s+)(?P<value>{_NAME_X})"
)

_NUM_SUFFIX = r"(?:\s*(?:number|num|no\.?|#|id))?"


def _cue_id_re(cues: str, value: str = r"[A-Za-z0-9][A-Za-z0-9-]{3,}") -> re.Pattern[str]:
    """`<cue> [number|no|#|id] [is|:|#] <value>`, with the cue matched case-insensitively."""
    return re.compile(
        rf"(?i:(?P<cue>\b(?:{cues})(?!\w){_NUM_SUFFIX}\s*(?:is|was|:|=|-)?\s*#?\s*))"
        rf"(?P<value>{value})"
    )


# (kind, pattern) in the order they run. Every value must contain a digit. Plates run
# before licences so "license plate 7ABC123" is a vehicle, not a licence.
STRICT_CUE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "VEHICLE",
        _cue_id_re(
            r"licen[cs]e\s*plate|number\s*plate|plate|registration|reg\.?|vin",
            r"[A-Z0-9]{1,4}[ -]?[A-Z0-9]{1,4}(?:[A-Z0-9]{9})?(?![\w-])",
        ),
    ),
    (
        "MRN",
        _cue_id_re(
            r"mrn|medical\s*records?|med(?:ical)?\s*rec(?:ord)?|chart|patient|hospital|encounter"
        ),
    ),
    (
        "HEALTH_PLAN",
        _cue_id_re(
            r"member|subscriber|beneficiary|health\s*plan|insurance|insurer|policy(?:holder)?|"
            r"group|medicare|medicaid|mbi|hicn"
        ),
    ),
    ("ACCOUNT", _cue_id_re(r"bank\s*account|account|acct\.?|routing")),
    (
        "LICENSE",
        _cue_id_re(
            r"(?:driver'?s?|driving|medical|professional|nursing|business)?\s*licen[cs]e|"
            r"dl|dea|npi|certificate|cert\.?|passport|permit"
        ),
    ),
    (
        "DEVICE",
        _cue_id_re(r"serial|s/n|sn|imei|meid|esn|udid|service\s*tag|device(?:\s*id)?"),
    ),
)
VIN_RE = re.compile(r"\b[A-HJ-NPR-Z0-9]{17}\b")
# A token with 6 or more digits anywhere in it ("88812345", "A1B2C3D4E5F6", "123-456-789"),
# not preceded by a currency sign. Hex error codes ("0x80070005") are excluded by a check.
LONG_ID_RE = re.compile(
    r"(?<![\w$€£¥.-])(?=(?:[A-Za-z-]*\d){6})[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?(?![\w-]|\.\w)"
)


# ---- helpers -------------------------------------------------------------------------------


def _compile_keep(keep_patterns: KeepPatterns | None, profile: str) -> list[re.Pattern[str]]:
    """The keep patterns to use: the caller's, else the profile's default."""
    if keep_patterns is None:
        keep_patterns = DEFAULT_KEEP_PATTERNS if profile == "strict" else ()
    return [p if isinstance(p, re.Pattern) else re.compile(p) for p in keep_patterns]


def _protected_spans(text: str, keep: Sequence[re.Pattern[str]]) -> list[tuple[int, int]]:
    """Sorted, merged (start, end) spans no rule may touch: existing placeholders and
    anything matching a keep pattern."""
    spans = [m.span() for m in PLACEHOLDER_RE.finditer(text)]
    for rx in keep:
        spans.extend(m.span() for m in rx.finditer(text) if m.end() > m.start())
    spans.sort()
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _gaps(text: str, keep: Sequence[re.Pattern[str]]) -> Iterator[tuple[int, int, bool]]:
    """Walk `text` as (start, end, is_protected) pieces, in order."""
    last = 0
    for start, end in _protected_spans(text, keep):
        if start > last:
            yield last, start, False
        yield start, end, True
        last = end
    if last < len(text):
        yield last, len(text), False


def _sub_outside(
    pattern: re.Pattern[str],
    text: str,
    repl: Callable[[re.Match[str]], str],
    keep: Sequence[re.Pattern[str]] = (),
) -> str:
    """`pattern.sub(repl, text)`, but never inside placeholders or kept values.

    Each unprotected piece is substituted on its own, so a pattern can never match
    across (or into) a protected value.
    """
    out = []
    for start, end, protected in _gaps(text, keep):
        piece = text[start:end]
        out.append(piece if protected else pattern.sub(repl, piece))
    return "".join(out)


def _finditer_outside(
    pattern: re.Pattern[str], text: str, keep: Sequence[re.Pattern[str]] = ()
) -> Iterator[tuple[int, int, re.Match[str]]]:
    """Like `pattern.finditer(text)` outside protected spans; yields (start, end, match)
    with offsets into the whole text."""
    for start, end, protected in _gaps(text, keep):
        if not protected:
            for m in pattern.finditer(text[start:end]):
                yield start + m.start(), start + m.end(), m


# An agent asking for a name: the customer's next turn is often just the name
# ("may I have your name please?" -> "Crystal Minh"), with no cue in that turn.
NAME_REQUEST_RE = re.compile(
    r"(?i)\b(?:(?:your|the)\s+(?:full\s+|first\s+(?:and\s+last\s+)?|last\s+)?name|you\s+(?:full|first|last)\s+name|"
    r"who\s+(?:am\s+i|i'?m)\s+(?:speaking|chatting|talking)\s+(?:to|with)|name\s+(?:please|on\s+the\s+account))"
)
# A reply that starts with a name: 1-4 name-like words, then the end of the turn or a
# comma/full stop ("Crystal Minh" / "Joseph Banter, I don't have my account number").
NAME_ONLY_REPLY_RE = re.compile(
    r"^\s*(?:(?:sure|yes|ok(?:ay)?|hi|hello)[.,!]?\s+)?"
    r"(?:(?:it'?s|it\s+is|i'?m|i\s+am|this\s+is|name\s+is)\s+)?"
    r"(?P<value>[A-Za-z][a-z'’-]+(?:\s+[A-Za-z][a-z'’-]+){0,3}?)\s*(?:[.!,;]|\s+and\b|$)",
    re.IGNORECASE,
)
# Usernames / logins / handles after a cue ("my username is josephb9", "Username: cminh730").
USERNAME_CUE_RE = re.compile(
    r"(?i)(?P<cue>\b(?:user\s?name|user\s?id|login\s+(?:name|id))\b(?:\s+is)?\s*[:=-]?\s*)"
    # a username must contain a digit: "username is josephb9", never "login issues"
    r"(?P<value>(?=[A-Za-z0-9._-]*\d)[A-Za-z0-9][A-Za-z0-9._-]{2,39})\b"
)
USERNAME_REQUEST_RE = re.compile(r"(?i)\b(?:user\s?name|user\s?id|account\s?id)\b")
# A handle inside a longer reply: starts with a letter, has a digit, not part of an
# email address or a URL ("cminh730", "j.doe_88"; never "cminh730@email.com").
HANDLE_IN_REPLY_RE = re.compile(
    r"(?<![\w@./:-])(?=[A-Za-z0-9._-]*\d)[A-Za-z][A-Za-z0-9._-]{2,39}(?<![._-])(?![\w@/:-]|\.\w)"
)
# A reply that is (mostly) one username-like token: letters+digits, no spaces.
USERNAME_REPLY_RE = re.compile(
    r"^\s*(?:(?:it'?s|my\s+username\s+is|username:?)\s*)?(?P<value>(?=[^\s,]*\d)[A-Za-z][A-Za-z0-9._-]{2,39})\s*(?:[.,;]|$)",
    re.I,
)


def _clean_name(value: str) -> str | None:
    """Trim a captured name at the first stopword; None if nothing name-like is left."""
    words: list[str] = []
    for w in value.split():
        if w.lower().strip("'’") in _NAME_STOP:
            break
        words.append(w)
    name = " ".join(words).strip(" ,.'’-")
    return name if len(name) >= 2 else None


def _plausible_name(value: str, *, require_caps: bool = True) -> str | None:
    """Stricter name check for NER and title cues: drop leading/trailing common words
    ("Will Smith" -> "Smith", "Name" -> None) and reject anything with digits.
    `require_caps=False` accepts lower-case names (tier-2 spans, already filtered)."""
    if any(ch.isdigit() for ch in value) or "@" in value or "<" in value:
        return None
    words = [w.strip(",.;:()'’\"") for w in value.split()]
    words = [w for w in words if w]
    while words and words[0].lower() in COMMON_WORDS:
        words.pop(0)
    while words and words[-1].lower() in COMMON_WORDS:
        words.pop()
    if not words or (require_caps and not all(w[0].isupper() for w in words)):
        return None
    name = " ".join(words)
    return name if len(name) >= 2 else None


def _parse_date(value: str) -> tuple[int | None, int | None, int | None]:
    """Best-effort (year, month, day) for a date string; None for missing parts.

    Numeric dates are read as month/day unless the first number is over 12 (then
    day/month) — ambiguous, but only used to keep placeholders consistent.
    """
    v = value.strip()
    if m := re.match(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", v):
        return int(m[1]), int(m[2]), int(m[3])
    if m := re.fullmatch(r"(\d{4})-(\d{1,2})", v):
        return int(m[1]), int(m[2]), None
    if m := re.fullmatch(r"(\d{1,2})/(\d{4})", v):
        return int(m[2]), int(m[1]), None
    if m := re.match(r"(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})$", v):
        a, b, y = int(m[1]), int(m[2]), int(m[3])
        if y < 100:
            y += 1900 if y > _THIS_YEAR % 100 else 2000
        month, day = (b, a) if a > 12 else (a, b)
        return y, month, day
    month = None
    if mm := re.search(_MONTH, v, re.I):
        month_names = "jan feb mar apr may jun jul aug sep oct nov dec".split()
        month = month_names.index(mm.group(0)[:3].lower()) + 1
    year_m = re.search(r"\b(\d{4})\b", v)
    year = int(year_m[1]) if year_m else None
    day_m = re.search(r"(?<!\d)(\d{1,2})(?:st|nd|rd|th)?(?!\d)", v)
    day = int(day_m[1]) if day_m else None
    return year, month, day


def _date_ok(value: str) -> bool:
    """True if the parsed month and day are in range."""
    year, month, day = _parse_date(value)
    if month is None or not 1 <= month <= 12:
        return False
    return day is None or 1 <= day <= 31


def _year_reveals_age_over_89(year: int) -> bool:
    """Safe Harbor: a year that implies someone is over 89 must go too."""
    return _THIS_YEAR - year >= 90


def _zip3(value: str) -> str:
    """'02139' or '02139-4307' -> '021XX'; restricted prefixes -> '000XX'."""
    prefix = value[:3]
    return ("000" if prefix in RESTRICTED_ZIP3 else prefix) + "XX"


# ---- surrogates ----------------------------------------------------------------------------

# Built-in name lists for surrogates: culturally diverse, many of them used for any gender
# (the text's pronouns are kept, so a neutral name clashes with them less often). None is
# an ordinary English word, so a surrogate never reads like part of the sentence.
SURROGATE_FIRST_NAMES: tuple[str, ...] = tuple(
    """
    Aarav Abebe Adaeze Adebayo Adrian Aiko Akira Alani Alexis Amani Amara Ananya Andrei
    Anika Ariel Arjun Asha Avery Ayaan Ayumi Bao Beatriz Bilal Bo Bongani Callum Camila
    Carmen Cheng Chiara Chidi Chioma Cyrus Dalia Dang Darius Daniela Dara Dmitri Eamon
    Elif Emeka Emerson Emil Esra Ezra Farah Fatima Femi Finley Freya Gabriel Giulia
    Hana Haruto Hassan Hiroshi Ibrahim Idris Ifeoma Ilse Imani Ines Ingrid Ishaan Jae
    Jalen Jamie Jaya Jin Joaquin Jordan Jun Kaito Kamala Kari Kasim Kavya Keanu Kenji
    Kiran Kofi Kwame Lars Laila Leilani Lena Leon Linh Lorenzo Luca Lucia Lukas Maeve
    Mai Malik Manon Marisol Mateo Meera Mehmet Mika Milan Min Mira Nadia Naveen Nia
    Nikhil Niko Noor Nuri Oksana Olu Omar Oren Paolo Parisa Priya Quinn Rafael Rahul
    Ravi Reza Riley Rina Rohan Ronan Rumi Rustam Sacha Sadia Samir Sana Santiago Sasha
    Selin Seo Shira Siddharth Simone Siobhan Soren Sunil Tamar Tariq Teodor Thandiwe
    Tomasz Tuan Uma Valentina Vikram Wen Xiomara Yara Yaw Yohannes Yuki Yusuf Zainab
    Zara Zeynep Zhen Zoltan
    """.split()
)
SURROGATE_LAST_NAMES: tuple[str, ...] = tuple(
    """
    Abara Abdullah Acheampong Adeyemi Agarwal Ahmadi Akhtar Alvarado Amador Andersen
    Antonelli Aquino Arslan Asante Baptiste Barbosa Bergstrom Bianchi Bose Brennan
    Cabrera Castellanos Chandra Chaudhry Chen Cisneros Costa Dasgupta Delgado Demir
    Desai Dimitrov Dlamini Duarte Eklund Esposito Estrada Farouk Fernandes Figueroa
    Fujita Gallardo Ghosh Gonzaga Gupta Haddad Hakimi Halvorsen Hashemi Herrera Horvath
    Hosseini Ibarra Ikeda Iqbal Ivanova Jankowski Janssen Jaramillo Jovanovic Kaur
    Kapoor Kariuki Kasprzak Kawamura Kelleher Khalil Kimura Kovacs Kowalczyk Kumar
    Lachance Lindqvist Lombardi Lozano Mahlangu Majumdar Malhotra Mbeki Medina Mehta
    Mendoza Mirza Moreau Mukherjee Murakami Nakamura Naidoo Nair Narang Navarro Ndlovu
    Nguyen Nikolaidis Novak Nwosu Obi Ochoa Odhiambo Ogunleye Okafor Okonkwo Oliveira
    Onyango Orozco Ortega Osei Ozturk Pacheco Papadopoulos Pereira Petrovic Pham
    Quintero Rahman Rajan Ramaswamy Rasmussen Reyes Rinaldi Rocha Rossi Saavedra Sahin
    Salazar Santos Sato Schreiber Sekar Serrano Shah Shapiro Siddiqui Silva Sokolov
    Solberg Suarez Suzuki Takahashi Tanaka Tavares Tembo Thakur Toure Tran Trujillo
    Uchenna Valdez Vargas Varma Vasquez Veloso Volkov Wanjiru Watanabe Yamamoto Yilmaz
    Yoshida Zamora Zapata Zielinski Zuniga
    """.split()
)
SURROGATE_STREETS: tuple[str, ...] = tuple(
    """
    Amberleaf Birchmoor Brambleton Copperfield Driftwood Elderberry Fernhollow Foxglove
    Glenhaven Hawthorne Heronwood Juniper Kestrel Larkspur Linden Marigold Meadowlark
    Millbrook Oakmere Pinecrest Quillfeather Ravenswood Rosewater Silverbirch Stonebridge
    Sycamore Thistledown Willowmere Windmere Wrenfield
    """.split()
)
SURROGATE_SUFFIXES: tuple[str, ...] = ("St", "Ave", "Rd", "Ln", "Dr", "Ct", "Way", "Pl")
# Town names that exist in many US states, so none points at one real place.
SURROGATE_CITIES: tuple[str, ...] = tuple(
    """
    Springfield Fairview Franklin Greenville Clinton Georgetown Salem Madison Arlington
    Ashland Burlington Centerville Kingston Marion Milford Newport Oxford Riverside
    Winchester Clayton Auburn Bristol Dover Hudson Monroe Jackson Lexington Florence
    """.split()
)
SURROGATE_EMAIL_DOMAINS: tuple[str, ...] = ("example.com", "example.org", "example.net")
# IPv4 documentation ranges (RFC 5737); IPv6 documentation prefix 2001:db8::/32 (RFC 3849)
SURROGATE_IPV4_NETS: tuple[str, ...] = ("192.0.2", "198.51.100", "203.0.113")

# Kinds that keep their placeholder even with surrogates (see the module docstring).
NO_SURROGATE_KINDS: frozenset[str] = frozenset({"DATE", "DOB", "NRP"})

_SURROGATE_MARK_RE = re.compile(r"<SURR_(\d+)>")


def _case_style(value: str) -> str:
    """'lower', 'upper' or 'title' — how the original was written."""
    letters = [ch for ch in value if ch.isalpha()]
    if letters and all(ch.islower() for ch in letters):
        return "lower"
    if len(letters) > 1 and all(ch.isupper() for ch in letters):
        return "upper"
    return "title"


def _apply_case(value: str, style: str) -> str:
    return value.lower() if style == "lower" else value.upper() if style == "upper" else value


def _pour_digits(template: str, digits: str) -> str:
    """Write `digits` into the digit positions of `template`, keeping everything else:
    _pour_digits("(977) 625-2661", "4155550137") == "(415) 555-0137"."""
    it = iter(digits)
    return "".join(next(it, "0") if ch.isdigit() else ch for ch in template)


def _luhn_complete(prefix: str) -> str:
    """Append the check digit that makes `prefix` Luhn-valid."""
    for check in "0123456789":
        if luhn_ok(prefix + check):
            return prefix + check
    raise AssertionError("unreachable: one check digit always works")


def _shape_like(value: str, rng: random.Random) -> str:
    """Same shape: each letter -> a random letter of the same case, digit -> digit."""
    out = []
    for ch in value:
        if ch.isdigit():
            out.append(rng.choice("0123456789"))
        elif ch.isalpha() and ch.isascii():
            letters = "ABCDEFGHJKLMNPRSTUVWXYZ"
            out.append(rng.choice(letters) if ch.isupper() else rng.choice(letters).lower())
        else:
            out.append(ch)
    return "".join(out)


def _random_digits(rng: random.Random, n: int, *, first_nonzero: bool = True) -> str:
    if n <= 0:
        return ""
    head = rng.choice("123456789") if first_nonzero else rng.choice("0123456789")
    return head + "".join(rng.choice("0123456789") for _ in range(n - 1))


def _fake_phone_digits(digits: str, international: bool, rng: random.Random) -> str:
    """A fictional number with as many digits as the original: NANP numbers use the
    555-0100..555-0199 range reserved for fiction; others end in 55501xx."""
    n = len(digits)
    line = "01" + _random_digits(rng, 2, first_nonzero=False)
    if n == 10 and not international:
        return rng.choice("23456789") + _random_digits(rng, 2, first_nonzero=False) + "555" + line
    if n == 11 and digits[0] == "1":
        return (
            "1"
            + rng.choice("23456789")
            + _random_digits(rng, 2, first_nonzero=False)
            + "555"
            + line
        )
    if n == 7:
        return "555" + line
    keep = digits[:2] if international and n > 9 else ""  # keep the country code
    tail = "555" + line if n - len(keep) >= 7 else ""
    middle = n - len(keep) - len(tail)
    return keep + _random_digits(rng, middle) + tail


def _fake_iban(value: str, rng: random.Random) -> str:
    """Same country code and layout, random account part, valid mod-97 check digits."""
    compact = re.sub(r"\s+", "", value).upper()
    country, bban = compact[:2], _shape_like(compact[4:], rng).upper()
    numeric = "".join(str(int(ch, 36)) for ch in bban + country + "00")
    check = f"{98 - int(numeric) % 97:02d}"
    fake = country + check + bban
    out, it = [], iter(fake)
    for ch in value:
        out.append(ch if ch.isspace() else next(it, ""))
    return "".join(out)


def _handle_from(first: str, last: str, original: str, fake_digits: str) -> str:
    """A login handle built from a surrogate name in the original's style: "j.doe" ->
    "i.okafor", "jane_doe" -> "imani_okafor", "cminh730" -> "iokafor418" (`fake_digits`
    replace the original's digits)."""
    f, l_ = first.lower(), last.lower()
    letters = re.sub(r"[^A-Za-z]", "", original)
    sep = re.search(r"[A-Za-z]([._-])[A-Za-z]", original)
    if sep:
        head = original.split(sep.group(1))[0]
        base = f"{f if len(head) > 1 else f[0]}{sep.group(1)}{l_}"
    elif len(letters) > 8:
        base = f + l_
    else:
        base = f[0] + l_
    return base + fake_digits


class _Surrogates:
    """Fake-value factory for one `RedactionSession` (see "Surrogates" in the module
    docstring). It never sees more than the session already holds: the salt, the
    placeholder of a value and its salted digest, plus the value itself at the moment it
    is being replaced (to copy its format and to make sure the fake differs from it).
    """

    def __init__(self, session: RedactionSession) -> None:
        self.s = session
        self.marks: list[str | None] = []  # marker index -> surrogate (once rendered)
        self.mark_ph: list[str] = []  # marker index -> placeholder (for mapping_hashes)
        self._makers: list[Callable[[], str]] = []
        self._mark_ids: dict[tuple[str, Any], int] = {}
        self.people: dict[str, tuple[str, str]] = {}  # NAME placeholder -> (first, last)
        self.base: dict[str, str] = {}  # placeholder -> base surrogate (per value)
        self.used: dict[str, set[str]] = {}  # kind -> surrogates handed out (lower-case)
        self.attempt = 0  # retry number inside `_pick`

    # -- deterministic randomness --
    def rng(self, ph: str, attempt: int = 0) -> random.Random:
        """A random generator seeded by the salt and the value's digest (+ retry no.)."""
        digest = self.s._ph_digest.get(ph, ph)
        seed = hashlib.sha256(self.s.salt + f"surrogate\x00{digest}\x00{attempt}".encode())
        return random.Random(int.from_bytes(seed.digest()[:8], "big"))

    def handle_digits(self, original: str, attempt: int) -> str:
        """Fake digits for a handle, seeded by the salted digest of the original's digits
        — so "cminh730" and "cminh730@email.com" get the same new digits."""
        digits = re.sub(r"\D", "", original)
        if not digits:
            return ""
        for extra in range(20):
            rng = self.rng(self.s._digest("HANDLE_DIGITS", digits), attempt * 20 + extra)
            fake = _random_digits(rng, len(digits))
            if fake != digits:
                return fake
        return fake

    def _fresh(self, kind: str, candidate: str) -> bool:
        """Not handed out before for this kind, and not equal to any original value of
        this kind seen in the session (checked through the salted digests)."""
        key = candidate.strip().lower()
        if key in self.used.get(kind, set()):
            return False
        return self.s._digest(kind, candidate) not in self.s._by_hash

    def _claim(self, kind: str, candidate: str) -> str:
        self.used.setdefault(kind, set()).add(candidate.strip().lower())
        return candidate

    def _pick(self, kind: str, ph: str, make: Callable[[random.Random], str]) -> str:
        """Generate until fresh (bounded; a 50th collision in a row is practically
        impossible, and the last candidate is used then). `make` may read
        `self.attempt` (the retry number) for parts that are not seeded by `ph`."""
        candidate = ""
        for attempt in range(50):
            self.attempt = attempt
            candidate = make(self.rng(ph, attempt))
            if self._fresh(kind, candidate):
                break
        return self._claim(kind, candidate)

    # -- people --
    def person(self, ph: str) -> tuple[str, str]:
        """The surrogate (first, last) for a NAME placeholder."""
        if ph in self.people:
            return self.people[ph]
        real = {k for k, v in self.s._names.items() if v == ph}
        real_parts = {w for k in real for w in k.split()}
        taken = {w for p in self.people.values() for w in (p[0].lower(), p[1].lower())}
        first, last = "", ""
        for attempt in range(50):
            rng = self.rng(ph, attempt)
            first, last = rng.choice(SURROGATE_FIRST_NAMES), rng.choice(SURROGATE_LAST_NAMES)
            words = {first.lower(), last.lower()}
            full = f"{first} {last}".lower()
            if not (words & (real_parts | taken | set(self.s._names))) and self._fresh(
                "NAME", full
            ):
                break
        self._claim("NAME", f"{first} {last}")
        self.people[ph] = (first, last)
        return first, last

    def primary_person(self) -> tuple[str, str]:
        """The conversation's first-named person (usually the customer), for usernames
        and emails; a person seeded from the session salt if no name was found."""
        name_phs = [ph for ph in self.s._ph_digest if ph.startswith("<NAME_")]
        if name_phs:
            return self.person(name_phs[0])
        return self.person("<NAME_0>")  # not a real placeholder: just a stable seed

    def render_name(self, ph: str, form: str, style: str) -> str:
        first, last = self.person(ph)
        value = {"first": first, "last": last}.get(form, f"{first} {last}")
        return _apply_case(value, style)

    # -- everything else --
    def _once(self, key: str, kind: str, ph: str, make: Callable[[random.Random], str]) -> str:
        """The surrogate stored under `key`, generated (fresh) on first use."""
        if key not in self.base:
            self.base[key] = self._pick(kind, ph, make)
        return self.base[key]

    def render(self, kind: str, ph: str, value: str) -> str:
        """The surrogate text for `value` (of `kind`, placeholder `ph`) in its format.

        Digits-only kinds (PHONE, CARD, SSN) pick one fake digit string per value and
        pour it into each spelling ("(977) 625-2661" and "977.625.2661" stay one number).
        """
        if kind == "PHONE":
            digits = re.sub(r"\D", "", value)
            intl = value.lstrip().startswith(("+", "00"))
            fake = self._once(ph, kind, ph, lambda r: _fake_phone_digits(digits, intl, r))
            return _pour_digits(value, fake)
        if kind == "CARD":
            n = len(re.sub(r"\D", "", value))

            def card(r: random.Random) -> str:
                return _luhn_complete("4111" + _random_digits(r, n - 5, first_nonzero=False))

            return _pour_digits(value, self._once(ph, kind, ph, card))
        if kind == "SSN":

            def ssn(r: random.Random) -> str:
                # 987-65-4320..4329 are reserved for advertising; then never-issued 9xx
                if len(self.used.get(kind, ())) < 5:
                    return "98765432" + r.choice("0123456789")
                return "9" + _random_digits(r, 8, first_nonzero=False)

            return _pour_digits(value, self._once(ph, kind, ph, ssn))
        if kind == "IP":

            def ip(r: random.Random) -> str:
                if ":" in value:
                    return "2001:db8::" + ":".join(f"{r.randrange(1, 0xFFFF):x}" for _ in "ab")
                return f"{r.choice(SURROGATE_IPV4_NETS)}.{r.randrange(1, 255)}"

            return self._once(ph, kind, ph, ip)
        if kind == "EMAIL":
            local = value.split("@", 1)[0]
            first, last = self.primary_person()

            def email(r: random.Random) -> str:
                handle = _handle_from(first, last, local, self.handle_digits(local, self.attempt))
                return f"{handle}@{r.choice(SURROGATE_EMAIL_DOMAINS)}"

            return self._once(ph, kind, ph, email)
        if kind == "USERNAME":
            first, last = self.primary_person()
            return self._once(
                ph,
                kind,
                ph,
                lambda r: _handle_from(first, last, value, self.handle_digits(value, self.attempt)),
            )
        if kind == "ADDRESS":
            style = _case_style(value)
            fake = self._once(
                f"{ph}\x00{style}", kind, ph, lambda r: self._address(value, r, self.s.strict)
            )
            return _apply_case(fake, style)
        if kind == "LOCATION":
            city = self._once(ph, kind, ph, lambda r: r.choice(SURROGATE_CITIES))
            return _apply_case(city, _case_style(value))
        if kind == "URL":
            scheme = "http://" if value.lower().startswith("http://") else "https://"

            def url(r: random.Random) -> str:
                path = _shape_like("abcd1234", r).lower()
                return f"{scheme}{r.choice(SURROGATE_EMAIL_DOMAINS)}/{path}"

            return self._once(ph, kind, ph, url)
        if kind == "IBAN":
            return self._once(ph, kind, ph, lambda r: _fake_iban(value, r))
        if kind == "MAC":

            def mac(r: random.Random) -> str:
                fake = "".join(f"{r.randrange(16):x}" if ch.isalnum() else ch for ch in value)
                return fake[0] + "2" + fake[2:]  # "x2:…" is locally administered: no vendor's

            return self._once(ph, kind, ph, mac)
        # IDs, MRN, ACCOUNT, LICENSE, DEVICE, VEHICLE, custom kinds: the same shape
        shape = re.sub(r"[A-Za-z]", "a", re.sub(r"\d", "9", value))
        return self._once(f"{ph}\x00{shape}", kind, ph, lambda r: _shape_like(value, r))

    @staticmethod
    def _address(value: str, r: random.Random, strict: bool) -> str:
        """A fictional address with the parts the original had: house number (same
        length), street (+ the original's suffix word), unit, town, state + ZIP."""
        number = re.search(r"\b(\d{1,5})[A-Za-z]?\s", value)
        suffix = re.search(rf"\b({_STREET})\b\.?", value, re.I)
        street = (
            (_random_digits(r, len(number.group(1))) if number else str(r.randrange(10, 9999)))
            + f" {r.choice(SURROGATE_STREETS)} "
            + (suffix.group(1).title() if suffix else r.choice(SURROGATE_SUFFIXES))
        )
        unit = re.search(r"(?i)\b(apt|apartment|suite|ste|unit|flat|#)\.?\s*([\w-]+)", value)
        if unit:
            street += (
                f", {unit.group(1).title()} {_shape_like(unit.group(2), r).lstrip('0') or '1'}"
            )
        parts = [street]
        zip_m = re.search(r"\b\d{5}(?:-\d{4})?\b(?!.*\d)", value)
        rest = value[unit.end() :] if unit else value
        if zip_m or re.search(r",\s*[A-Za-z]{3,}", rest):
            parts.append(r.choice(SURROGATE_CITIES))
        if zip_m:
            fake_zip = _random_digits(r, 5)
            state = r.choice([s for s in _STATE_ABBR if s != "PR"])
            parts.append(f"{state} {_zip3(fake_zip) if strict else fake_zip}")  # strict: ZIP3
        return ", ".join(parts)

    # -- markers --
    def mark(self, ph: str, variant: Any, make: Callable[[], str]) -> str:
        """An internal `<SURR_n>` marker standing for one rendered surrogate.

        Markers look like placeholders, so every later rule leaves them alone; the
        session swaps them for their text at the very end (`RedactionSession.render`).
        The fake value itself is only generated then, once every name of the
        conversation is known (so it can be checked against all of them).
        """
        key = (ph, variant)
        if key not in self._mark_ids:
            self._mark_ids[key] = len(self.marks)
            self.marks.append(None)
            self._makers.append(make)
            self.mark_ph.append(ph)
        return f"<SURR_{self._mark_ids[key]}>"

    def resolve(self, index: int) -> str:
        """The surrogate text for marker `index` (generated on first use)."""
        value = self.marks[index]
        if value is None:
            value = self.marks[index] = self._makers[index]()
        return value


# ---- the session ---------------------------------------------------------------------------


class RedactionSession:
    """Placeholder state for one conversation (or one exported object).

    Use one session per conversation so placeholders restart at 1 and the same value
    always maps to the same placeholder within it. With `surrogates=True` the output
    carries consistent fake values instead of placeholders (see the module docstring).
    """

    def __init__(
        self,
        *,
        salt: bytes | None = None,
        pseudonymize_ids: bool = False,
        redact_private_ips: bool = False,
        extra_patterns: ExtraPatterns | None = None,
        profile: Profile = "standard",
        keep_patterns: KeepPatterns | None = None,
        surrogates: bool = False,
    ) -> None:
        if profile not in PROFILES:
            raise ValueError(f"unknown redaction profile {profile!r} (use one of {PROFILES})")
        self.profile = profile
        self.strict = profile == "strict"
        self.salt = salt if salt is not None else secrets.token_bytes(16)
        self.pseudonymize_ids = pseudonymize_ids
        self.redact_private_ips = redact_private_ips or self.strict
        self.keep = _compile_keep(keep_patterns, profile)
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
        # other values the NER layer found (exact text -> kind), e.g. "Boston" -> LOCATION
        self._known: dict[str, str] = {}
        self._known_re: re.Pattern[str] | None = None
        # how each registered name (or part) is used: "full", "first" or "last", so a
        # surrogate can render "Crystal" as the surrogate person's first name
        self._name_forms: dict[str, str] = {}
        self._name_form_from_full: set[str] = set()
        # usernames / handles for this conversation (cross-turn cues, tier 2)
        self.usernames: set[str] = set()
        self._ph_digest: dict[str, str] = {}  # placeholder -> full salted digest
        self.surrogates = surrogates and profile != "off"
        self._sur: _Surrogates | None = _Surrogates(self) if self.surrogates else None

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
            self._ph_digest[ph] = digest
        if count:
            self.counts[kind] += 1
        return ph

    def _is_kept(self, value: str) -> bool:
        return any(rx.fullmatch(value) for rx in self.keep)

    # -- output: placeholder or surrogate --
    def _out(self, kind: str, value: str, ph: str) -> str:
        """What to write for `value` (already allocated as `ph`): the placeholder
        itself, or — with surrogates — a marker for its fake value in this format."""
        if self._sur is None or kind in NO_SURROGATE_KINDS:
            return ph
        if kind == "NAME":
            return self._name_out(ph, value)
        sur = self._sur
        variant = re.sub(r"[A-Za-z]", "a", re.sub(r"\d", "9", value)) + _case_style(value)
        return sur.mark(ph, (kind, variant), lambda: sur.render(kind, ph, value))

    def _name_out(self, ph: str, matched: str) -> str:
        """Placeholder, or a marker for the surrogate person's full/first/last name,
        written in the same case as `matched`."""
        if self._sur is None:
            return ph
        sur = self._sur
        key = " ".join(matched.lower().split())
        form = self._name_forms.get(key) or ("full" if len(key.split()) > 1 else "first")
        style = _case_style(matched)
        return sur.mark(ph, ("NAME", form, style), lambda: sur.render_name(ph, form, style))

    def render(self, text: str) -> str:
        """Swap surrogate markers for their fake values (no-op without surrogates)."""
        if self._sur is None or "<SURR_" not in text:
            return text
        sur = self._sur
        return _SURROGATE_MARK_RE.sub(
            lambda m: sur.resolve(int(m[1])) if int(m[1]) < len(sur.marks) else m[0], text
        )

    # -- names --
    def add_name(
        self, name: str, *, careful: bool = False, lowercase_ok: bool = False
    ) -> str | None:
        """Register a person's name (and its parts) so every mention is redacted.
        Returns its placeholder, or None if it does not look like a name.

        `careful=True` (used for NER and title cues, which are noisier than "my name
        is") also rejects common words, so "Will" or "Patient Name" is not registered —
        registered names are matched case-insensitively everywhere. `lowercase_ok`
        (with `careful`, for tier-2 spans that already passed the false-positive
        filter) accepts names typed in lower case ("crystal minh").
        """
        cleaned = _clean_name(name)
        if cleaned and careful:
            cleaned = _plausible_name(cleaned, require_caps=not lowercase_ok)
        if not cleaned or self._is_kept(cleaned):
            return None
        key = cleaned.lower()
        words = key.split()
        self._note_name_forms(key, words)
        if key in self._names:
            return self._names[key]
        # "Jane Doe" after "Jane": same person, same placeholder
        known_part = next((self._names[p] for p in words if p in self._names), None)
        ph = known_part or self.placeholder("NAME", key, count=False)
        self._names[key] = ph
        for part in words:
            if len(part) >= 3 and part not in _NAME_STOP:
                if careful and part in COMMON_WORDS:
                    continue
                self._names.setdefault(part, ph)
        self._names_re = None
        return ph

    def _note_name_forms(self, key: str, words: list[str]) -> None:
        """Remember whether a name (part) is a full name, a first or a last name. A
        position inside a full name beats the guess made for a lone word."""
        if len(words) == 1:
            self._name_forms.setdefault(key, "first")
            return
        self._name_forms[key] = "full"
        for i, part in enumerate(words):
            if part in self._name_form_from_full:
                continue
            self._name_forms[part] = (
                "first" if i == 0 else "last" if i == len(words) - 1 else "first"
            )
            self._name_form_from_full.add(part)

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

    def collect_names(self, text: str) -> None:
        """Register every cue-introduced name in `text` (strict: titles too)."""
        for name in self.find_names(text):
            self.add_name(name)
        if self.strict:
            for rx in (NAME_TITLE_RE, NAME_LOOSE_GREETING_RE):
                for m in rx.finditer(text):
                    self.add_name(m.group("value"), careful=True)

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
            return self._name_out(self._names[m.group(1).lower()], m.group(1))

        return _sub_outside(self._names_re, text, repl, self.keep)

    # -- NER findings --
    def register_ner(self, text: str, spans: Sequence[ner_layer.NerSpan]) -> None:
        """Feed NER spans (found on the original `text`) into this session.

        People go into the same name registry as cue-found names; everything else is
        remembered as an exact value and replaced after the regex rules have run (so a
        span the regex already replaced is not replaced twice). Spans overlapping a
        kept value or an existing placeholder are ignored.
        """
        protected = _protected_spans(text, self.keep)
        for span in spans:
            if any(s < span.end and span.start < e for s, e in protected):
                continue
            value = span.text.strip()
            if len(value) < 2:
                continue
            if span.kind == "NAME":
                self.add_name(value, careful=True)
            elif span.kind == "USERNAME":
                self.usernames.add(value)
            elif span.kind == "DATE":
                # only real calendar dates; "yesterday", "Monday", "2024" stay
                if self.strict and re.search(_MONTH, value, re.I) and _date_ok(value):
                    self._known.setdefault(value, "DATE")
                    self._known_re = None
            elif span.kind in ("LOCATION", "NRP"):
                if value.lower() in COMMON_WORDS or not (value[0].isupper() or value[0].isdigit()):
                    continue
                self._known.setdefault(value, span.kind)
                self._known_re = None
            elif _has_digit(value):  # licence, passport, NHS, ITIN, bank numbers
                self._known.setdefault(value, span.kind)
                self._known_re = None

    def register_tier2(self, text: str, spans: Sequence[pii_model.Span]) -> None:
        """Feed tier-2 model spans (found on the original `text`, already through
        `pii_model.filter_spans`) into this session — the same registries as NER.

        Names may be lower case (the filter vetted them); usernames join the
        conversation's handles; emails, phones and addresses become exact values replaced
        after the regex rules (so nothing is replaced twice). Account numbers are only
        used in strict or with `pseudonymize_ids`: standard keeps record keys.
        """
        protected = _protected_spans(text, self.keep)
        for span in spans:
            if any(s < span.end and span.start < e for s, e in protected):
                continue
            value = span.text.strip()
            if len(value) < 2 or self._is_kept(value):
                continue
            if span.kind == "NAME":
                self.add_name(value, careful=True, lowercase_ok=True)
            elif span.kind == "USERNAME":
                self.usernames.add(value)
            elif span.kind in ("EMAIL", "PHONE", "ADDRESS"):
                self._known.setdefault(value, span.kind)
                self._known_re = None
            elif span.kind == "ACCOUNT" and (self.strict or self.pseudonymize_ids):
                self._known.setdefault(value, "ACCOUNT")
                self._known_re = None

    def _replace_known_values(self, text: str) -> str:
        if not self._known:
            return text
        if self._known_re is None:
            alts = sorted(self._known, key=len, reverse=True)
            self._known_re = re.compile(
                r"(?<![\w<])(" + "|".join(re.escape(a) for a in alts) + r")(?![\w>])"
            )

        def repl(m: re.Match[str]) -> str:
            value = m.group(1)
            kind = self._known[value]
            if kind == "DATE":
                return self._date_repl_value(value) or value
            return self._out(kind, value, self.placeholder(kind, value))

        return _sub_outside(self._known_re, text, repl, self.keep)

    # -- the pipeline --
    def _typed(self, kind: str, check: Callable[[str], bool] | None = None) -> Callable:
        """A re.sub callback replacing the `value` group (or the whole match)."""

        def repl(m: re.Match[str]) -> str:
            has_value = "value" in m.re.groupindex and m.group("value") is not None
            value = m.group("value") if has_value else m.group(0)
            if check is not None and not check(value):
                return m.group(0)
            ph = self._out(kind, value, self.placeholder(kind, value))
            if has_value:
                start, end = m.span("value")
                base = m.start()
                whole = m.group(0)
                return whole[: start - base] + ph + whole[end - base :]
            return ph

        return repl

    def _value_repl(self, make: Callable[[str], str | None]) -> Callable:
        """A re.sub callback replacing the `value` group with `make(value)` (None keeps
        the match). For rewrites that are not plain placeholders (ZIP3, "90+", dates)."""

        def repl(m: re.Match[str]) -> str:
            new = make(m.group("value"))
            if new is None:
                return m.group(0)
            start, end = m.span("value")
            whole, base = m.group(0), m.start()
            return whole[: start - base] + new + whole[end - base :]

        return repl

    def _url_check(self, url: str) -> bool:
        if self.strict:
            return True  # Safe Harbor: every URL
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
            ph = self._out("PHONE", value, self.placeholder("PHONE", value))
            return m.group(0).replace(m.group("value"), ph)
        return m.group(0)

    # -- strict-only rewrites --
    def _date_repl_value(self, value: str) -> str | None:
        """'May 1, 2024' -> '<DATE_1> 2024' (the year is allowed by Safe Harbor);
        a year 90+ years ago is dropped too; an invalid date is kept (None)."""
        if not _date_ok(value):
            return None
        year, month, day = _parse_date(value)
        key = f"{year}-{month}-{day}" if month else value
        ph = self.placeholder("DATE", key)
        if year is None or _year_reveals_age_over_89(year):
            return ph
        return f"{ph} {year}"

    def _birth_year_value(self, value: str) -> str | None:
        if not _year_reveals_age_over_89(int(value)):
            return None
        return self.placeholder("DATE", value)

    def _age_value(self, value: str) -> str:
        self.counts["AGE"] += 1
        return "90+"

    def _zip_value(self, value: str) -> str:
        self.counts["ZIP"] += 1
        return _zip3(value)

    def _long_id_check(self, value: str) -> bool:
        return not value.lower().startswith("0x")

    @staticmethod
    def _vin_check(value: str) -> bool:
        return _has_digit(value) and any(ch.isalpha() for ch in value)

    def _city_check(self, value: str) -> bool:
        return value.split()[0].lower() not in COMMON_WORDS

    def _plate_check(self, value: str) -> bool:
        return _has_digit(value) and any(ch.isalpha() for ch in value)

    def redact(
        self,
        text: str,
        ner_spans: Sequence[ner_layer.NerSpan] | None = None,
        *,
        render: bool = True,
    ) -> RedactionResult:
        """Redact one piece of text with this session's placeholders (or surrogates).

        `ner_spans` are NER findings for exactly this `text` (see
        `envforge.builder.ner.analyze_texts`); they are optional. `render=False` leaves
        surrogate markers in place (call `render` later); it changes nothing without
        surrogates. With profile "off" the text comes back unchanged.
        """
        if self.profile == "off":
            return RedactionResult(text=text)
        before = Counter(self.counts)
        keep = self.keep

        def sub(rx: re.Pattern[str], repl: Callable[[re.Match[str]], str]) -> None:
            nonlocal text
            text = _sub_outside(rx, text, repl, keep)

        if ner_spans:
            self.register_ner(text, ner_spans)
        for kind, rx in self.extra_patterns:
            sub(rx, self._typed(kind))
        sub(URL_RE, self._typed("URL", self._url_check))
        sub(EMAIL_RE, self._typed("EMAIL"))
        if self.strict:  # cue-based identifiers first, so the cue decides the type
            for kind, rx in STRICT_CUE_PATTERNS:
                check = self._plate_check if kind == "VEHICLE" else _id_value_ok
                sub(rx, self._typed(kind, check))
        sub(IBAN_RE, self._typed("IBAN", iban_ok))
        sub(CARD_RE, self._typed("CARD", self._card_check))
        sub(SSN_RE, self._typed("SSN"))
        sub(SSN_CUE_RE, self._typed("SSN"))
        sub(MAC_RE, self._typed("MAC"))
        sub(IPV6_RE, self._typed("IP", self._ip_check))
        sub(IPV4_RE, self._typed("IP", self._ip_check))
        sub(DOB_RE, self._typed("DOB"))
        if self.strict:
            sub(STRICT_DATE_RE, self._value_repl_whole(self._date_repl_value))
            sub(BIRTH_YEAR_RE, self._value_repl(self._birth_year_value))
            sub(AGE_SUFFIX_RE, self._value_repl(self._age_value))
            sub(AGE_CUE_RE, self._value_repl(self._age_value))
            sub(VIN_RE, self._typed("VEHICLE", self._vin_check))
        sub(PHONE_RE, self._phone_repl)
        sub(SHIP_CUE_RE, self._typed("ADDRESS"))
        sub(ADDRESS_CUE_RE, self._typed("ADDRESS", _has_digit))
        sub(ADDRESS_RE, self._typed("ADDRESS"))
        sub(UK_POSTCODE_RE, self._typed("ADDRESS"))
        if self.strict:
            sub(ZIP_STATE_RE, self._value_repl(self._zip_value))
            sub(ZIP_CUE_RE, self._value_repl(self._zip_value))
            sub(CITY_STATE_RE, self._typed("LOCATION", self._city_check))
        self.collect_names(text)
        text = self._replace_known_names(text)
        text = self._replace_known_values(text)
        if self.pseudonymize_ids:
            sub(_ID_CUE_RE, self._typed("ID", _has_digit))
            sub(_ID_STANDALONE_RE, self._typed("ID"))
        if self.strict:
            sub(LONG_ID_RE, self._typed("ID", self._long_id_check))
        if self.usernames:
            text = self._replace_usernames(text)
        counts = {k: v - before.get(k, 0) for k, v in self.counts.items() if v - before.get(k, 0)}
        used = set(PLACEHOLDER_RE.findall(text))
        if self._sur is not None:
            used |= {self._sur.mark_ph[int(i)] for i in _SURROGATE_MARK_RE.findall(text)}
        return RedactionResult(
            text=self.render(text) if render else text,
            counts_by_type=counts,
            mapping_hashes={ph: h for ph, h in self.hashes.items() if ph in used},
        )

    def _replace_usernames(self, text: str) -> str:
        """Replace every mention of the conversation's usernames (an account handle
        identifies a person as well as a name does). Counts real replacements only."""
        for u in sorted(self.usernames, key=len, reverse=True):
            if self._is_kept(u):
                continue
            rx = re.compile(rf"(?i)(?<![\w@.-]){re.escape(u)}(?![\w-])")

            def repl(m: re.Match[str], u: str = u) -> str:
                ph = self.placeholder("USERNAME", u, count=False)
                return self._out("USERNAME", m.group(0), ph)

            pieces = []
            hits = 0
            for start, end, protected in _gaps(text, self.keep):
                piece = text[start:end]
                if not protected:
                    piece, n = rx.subn(repl, piece)
                    hits += n
                pieces.append(piece)
            if hits:
                text = "".join(pieces)
                self.counts["USERNAME"] += hits
        return text

    @staticmethod
    def _value_repl_whole(make: Callable[[str], str | None]) -> Callable:
        """Like `_value_repl`, for patterns without a `value` group (the whole match)."""

        def repl(m: re.Match[str]) -> str:
            new = make(m.group(0))
            return m.group(0) if new is None else new

        return repl

    def redact_speaker(self, label: str | None, *, render: bool = True) -> str | None:
        """'Jane' -> '<NAME_1>', 'Agent Mike' -> 'Agent <NAME_2>', 'Customer' unchanged.
        (With surrogates: 'Imani', 'Agent Soren'; `render=False` leaves a marker.)"""
        if not label:
            return label
        words = label.split()
        role_words = [w for w in words if normalize_role(w) != "unknown"]
        name_words = [w for w in words if normalize_role(w) == "unknown"]
        name = " ".join(name_words).strip(" ()-–:")
        if not name:
            return label
        if self.profile == "off":
            return label
        ph = self.add_name(name) or self.placeholder("NAME", name, count=False)
        self.counts["NAME"] += 1
        out = self._name_out(ph, name)
        out = self.render(out) if render else out
        return " ".join([*role_words, out]) if role_words else out


# ---- public helpers ------------------------------------------------------------------------


def _ner_spans_for(
    texts: Sequence[str], *, ner: bool, profile: Profile, ner_model: str | None
) -> list[list[ner_layer.NerSpan]] | None:
    """NER spans per text when `ner` is on (empty lists if the layer is unavailable)."""
    if not ner or profile == "off":
        return None
    return ner_layer.analyze_texts(texts, profile=profile, model=ner_model)


def redact_text(
    text: str,
    *,
    pseudonymize_ids: bool = False,
    redact_private_ips: bool = False,
    extra_patterns: ExtraPatterns | None = None,
    known_names: list[str] | None = None,
    salt: bytes | None = None,
    profile: Profile = "standard",
    keep_patterns: KeepPatterns | None = None,
    ner: bool = False,
    ner_model: str | None = None,
    surrogates: bool = False,
) -> RedactionResult:
    """Redact a single piece of text (its own placeholder numbering).

    `extra_patterns` maps a type name to a regex, e.g.
    {"PASSPORT": r"\\b[A-Z]\\d{8}\\b"}; if the regex has a group named `value`, only
    that group is replaced. `known_names` are redacted wherever they appear.
    `profile`, `keep_patterns`, `ner` and `surrogates` are explained in the module
    docstring.
    """
    if profile == "off":
        return RedactionResult(text=text)
    session = RedactionSession(
        salt=salt,
        pseudonymize_ids=pseudonymize_ids,
        redact_private_ips=redact_private_ips,
        extra_patterns=extra_patterns,
        profile=profile,
        keep_patterns=keep_patterns,
        surrogates=surrogates,
    )
    for name in known_names or []:
        session.add_name(name)
    spans = _ner_spans_for([text], ner=ner, profile=profile, ner_model=ner_model)
    return session.redact(text, spans[0] if spans else None)


def _question_before(turns: Sequence[Turn], i: int) -> str:
    """The latest agent message within the 3 turns before turn `i` ("" if none).

    That is the question a customer's turn may be answering: customers often send
    another line ("What do you need from me?") before the answer itself, and system
    lines ("Searching the FAQ pages ...") can sit in between — they are skipped.
    """
    recent = reversed(turns[max(0, i - 3) : i])
    return next((p.text for p in recent if p.role not in ("customer", "system")), "")


def _mark_off(conv: Conversation) -> tuple[Conversation, dict[str, int]]:
    """Profile "off": the conversation unchanged, but marked as an explicit choice."""
    meta = {k: v for k, v in conv.meta.items() if k != "redaction_counts"}
    meta.update(
        redacted=True,
        redaction_counts={},
        redaction_profile="off",
        redaction_ner=False,
        redaction_tier2=False,
    )
    turns = [replace(t) for t in conv.turns]
    return replace(conv, turns=turns, meta=meta), {}


def _redact_one(
    conv: Conversation,
    *,
    profile: Profile,
    salt: bytes | None,
    surrogates: bool,
    ner_spans: Sequence[Sequence[ner_layer.NerSpan]] | None,
    ner_used: bool,
    tier2_spans: Sequence[Sequence[pii_model.Span]] | None = None,
    **session_options: Any,
) -> tuple[Conversation, dict[str, int], RedactionSession]:
    """Tier 1 (+ NER and tier-2 spans when given) for one conversation; also returns
    the session so tier-2 filtering can see which names were registered."""
    session = RedactionSession(salt=salt, profile=profile, surrogates=surrogates, **session_options)
    # speaker names are registered first (they get the lowest numbers) …
    speakers = [session.redact_speaker(t.speaker, render=False) for t in conv.turns]
    for i, t in enumerate(conv.turns):
        if ner_spans is not None:
            session.register_ner(t.text, ner_spans[i])
        if tier2_spans is not None:
            session.register_tier2(t.text, tier2_spans[i])
        session.collect_names(t.text)
        # cross-turn cues: the previous turn asked for a name / username and this turn
        # answers with one (no cue words in the answer itself)
        prev = _question_before(conv.turns, i)
        if prev and NAME_REQUEST_RE.search(prev):
            m = NAME_ONLY_REPLY_RE.match(t.text)
            if m and m.group("value").lower().split()[0] not in _NAME_STOP:
                session.add_name(m.group("value"))
        for m in USERNAME_CUE_RE.finditer(t.text):
            session.usernames.add(m.group("value"))
        if prev and USERNAME_REQUEST_RE.search(prev) and t.role != "system":
            m = USERNAME_REPLY_RE.match(t.text)
            if m:
                session.usernames.add(m.group("value"))
            elif len(t.text.split()) <= 12:
                # "davidwilliams617 davidwilliams617@email.com 0902260100": the reply
                # to "username, email and order ID?" — a handle among other values
                session.usernames.update(h.group(0) for h in HANDLE_IN_REPLY_RE.finditer(t.text))
    # … and only rendered once every name of the conversation is known
    turns = [
        Turn(role=t.role, text=session.redact(t.text).text, ts=t.ts,
             speaker=session.render(spk) if spk else spk)
        for t, spk in zip(conv.turns, speakers, strict=True)
    ]  # fmt: skip
    counts = dict(session.counts)
    meta = {k: v for k, v in conv.meta.items() if k not in ("redaction_counts",)}
    meta.update(
        redacted=True,
        redaction_counts=counts,
        redaction_profile=profile,
        redaction_ner=ner_used,
        redaction_surrogates=session.surrogates,
        redaction_tier2=tier2_spans is not None,
    )
    conv_id = conv.id
    if session.redact(conv.id).text != conv.id:  # an email or phone number used as the id
        conv_id = "conv-" + session._digest("ID", conv.id)[:12]  # salted: not reversible
        meta["id_redacted"] = True
    return replace(conv, id=conv_id, turns=turns, meta=meta), counts, session


def redact_conversation(
    conv: Conversation,
    *,
    pseudonymize_ids: bool = False,
    redact_private_ips: bool = False,
    extra_patterns: ExtraPatterns | None = None,
    salt: bytes | None = None,
    profile: Profile = "standard",
    keep_patterns: KeepPatterns | None = None,
    ner: bool = False,
    ner_model: str | None = None,
    ner_spans: Sequence[Sequence[ner_layer.NerSpan]] | None = None,
    surrogates: bool = False,
    tier2: Tier2Mode = "off",
    tier2_threshold: float = DEFAULT_TIER2_THRESHOLD,
    tier2_model: str | None = None,
    tier2_spans: Sequence[Sequence[pii_model.Span]] | None = None,
) -> tuple[Conversation, dict[str, int]]:
    """Redact every turn (text and speaker label) with one consistent session.

    Two passes: first collect names from speaker labels, name cues and NER across the
    whole conversation, then redact each turn — so "Hi Jane" in turn 1 is caught even
    if Jane only says "my name is Jane" in turn 5.

    `ner_spans` (one list per turn) lets a caller that already ran NER over many
    conversations in one batch pass the results in; normally leave it None and set
    `ner=True` instead. `tier2_spans` is the same for already-filtered tier-2 spans;
    normally use `tier2="auto"` (see `redact_conversations`) instead.

    Returns a new conversation (the input is not modified) with
    `meta["redacted"] = True`, `meta["redaction_counts"]`, `meta["redaction_profile"]`,
    `meta["redaction_ner"]` (whether NER actually ran), `meta["redaction_surrogates"]`
    and `meta["redaction_tier2"]`, plus the counts.
    """
    if profile not in PROFILES:
        raise ValueError(f"unknown redaction profile {profile!r} (use one of {PROFILES})")
    if profile == "off":
        return _mark_off(conv)
    options = dict(
        pseudonymize_ids=pseudonymize_ids,
        redact_private_ips=redact_private_ips,
        extra_patterns=extra_patterns,
        keep_patterns=keep_patterns,
    )
    if tier2 != "off" and tier2_spans is None:
        batch = redact_batch(
            [conv],
            profile=profile,
            ner=ner,
            ner_model=ner_model,
            surrogates=surrogates,
            tier2=tier2,
            tier2_threshold=tier2_threshold,
            tier2_model=tier2_model,
            ner_spans=[ner_spans] if ner_spans is not None else None,
            salt=salt,
            **options,
        )
        out = batch.conversations[0]
        return out, dict(out.meta["redaction_counts"])
    if ner_spans is None:
        ner_spans = _ner_spans_for(
            [t.text for t in conv.turns], ner=ner, profile=profile, ner_model=ner_model
        )
    ner_used = ner_spans is not None and bool(ner_layer.ner_available(ner_model))
    out, counts, _ = _redact_one(
        conv,
        profile=profile,
        salt=salt,
        surrogates=surrogates,
        ner_spans=ner_spans,
        ner_used=ner_used,
        tier2_spans=tier2_spans,
        **options,
    )
    return out, counts


def _replace_usernames(session: RedactionSession, text: str, usernames: set[str]) -> str:
    """Replace every mention of `usernames` in `text` with the session's placeholders.
    (Kept for callers of the old helper; sessions now do this in `redact`.)"""
    session.usernames |= set(usernames)
    return session.render(session._replace_usernames(text))


# ---- tier-2 triage -------------------------------------------------------------------------

# How much each signal adds to a conversation's risk score (see `risk`). With the
# default threshold of 1.0, one strong signal (an identity answer tier 1 left alone)
# is enough on its own; weak signals must add up.
RISK_WEIGHTS: dict[str, float] = {
    "name_answer_unredacted": 1.0,  # "may I have your name?" -> "crystal minh" kept
    "username_answer_unredacted": 1.0,  # "username?" -> "cminh730" kept
    "short_reply_after_identity_question": 0.5,  # "email / phone / account?" -> short reply kept
    "at_sign_remaining": 0.6,  # something@something survived
    "capitalised_pair": 0.5,  # "Joseph Banter" style bigram survived
    "letter_digit_token": 0.5,  # handle-like token ("jb1987") survived
    "long_digit_run": 0.2,  # 7+ digits survived (often an order number)
    "ner_off": 0.3,  # no NER net under the rules
}
IDENTITY_REQUEST_RE = re.compile(
    r"(?i)\b(?:name|user\s?name|user\s?id|account\s?(?:id|number|#)|e-?mail|phone|"
    r"(?:mobile|cell|contact)\s+number|address|zip(?:\s*code)?|postal\s+code|date\s+of\s+birth)\b"
)
_LETTER_DIGIT_RE = re.compile(
    r"(?<![\w<@.-])(?=[\w.-]*[A-Za-z])(?=[\w.-]*\d)[A-Za-z0-9][\w.-]{3,}(?![\w>@-])"
)
_NOT_A_HANDLE_RE = re.compile(
    r"(?i)^\d+(?:st|nd|rd|th|s|am|pm|gb|mb|kb|tb|k|m|x|hz|mhz|ghz|mbps|gbps|ms|min|hrs?|yrs?)$"
)
_LONG_DIGITS_RE = re.compile(r"(?<![\w<])\d{7,}(?![\w>])")
_AT_RE = re.compile(r"[\w.+-]+@[\w.-]+")


def _content_words(text: str) -> list[str]:
    """Words of a reply that are not filler ("sure", "ok", "it's", "my", …)."""
    words = [w.strip(".,!?;:'’\"()").lower() for w in text.split()]
    return [w for w in words if w and w not in COMMON_WORDS and w not in _NAME_STOP]


def risk(
    conv_before: Conversation,
    conv_after: Conversation,
    *,
    ner_used: bool | None = None,
    keep_patterns: KeepPatterns | None = None,
    profile: Profile = "standard",
) -> tuple[float, list[str]]:
    """How likely is it that tier 1 missed personal data in this conversation?

    Compares the original and the tier-1 output turn by turn (they must have the same
    turns) and returns `(score, reasons)`: the sum of `RISK_WEIGHTS` for every signal
    seen, and the signal names. Works with placeholder and surrogate output alike —
    a turn counts as "left alone" when it is unchanged, and a remnant only counts if
    the same text was already in the original (a surrogate never was). `ner_used`
    defaults to `conv_after.meta["redaction_ner"]`.
    """
    keep = _compile_keep(keep_patterns, profile)
    if ner_used is None:
        ner_used = bool(conv_after.meta.get("redaction_ner"))
    reasons: set[str] = set()
    before_turns, after_turns = conv_before.turns, conv_after.turns
    for i, (b, a) in enumerate(zip(before_turns, after_turns, strict=False)):
        original, out = b.text, a.text
        prev = _question_before(before_turns, i)
        # an agent line naming an identity field is a request, "?" or not ("Also
        # account id and order id"); only customer replies tier 1 left alone count
        if b.role == "customer" and prev and out == original:
            content = _content_words(original)
            n_words = len(original.split())
            if NAME_REQUEST_RE.search(prev) and content and n_words <= 6:
                if any(ch.isalpha() for ch in " ".join(content)):
                    reasons.add("name_answer_unredacted")
            if USERNAME_REQUEST_RE.search(prev) and n_words <= 6:
                if any(_LETTER_DIGIT_RE.search(w) for w in original.split()) or (
                    n_words == 1 and content
                ):
                    reasons.add("username_answer_unredacted")
            if IDENTITY_REQUEST_RE.search(prev) and content and n_words <= 4:
                reasons.add("short_reply_after_identity_question")

        def remaining(rx: re.Pattern[str], out: str = out, original: str = original) -> list[str]:
            """Matches in the output that were already in the original."""
            return [
                m.group(0) for _, _, m in _finditer_outside(rx, out, keep) if m.group(0) in original
            ]

        if remaining(_AT_RE):
            reasons.add("at_sign_remaining")
        for _, _, m in _finditer_outside(_NAME_BIGRAM_RE, out, keep):
            pair = m.group(0)
            if (
                pair in original
                and m.group(1).lower() not in COMMON_WORDS
                and m.group(2).lower() not in COMMON_WORDS
            ):
                reasons.add("capitalised_pair")
                break
        if any(not _NOT_A_HANDLE_RE.match(t) for t in remaining(_LETTER_DIGIT_RE)):
            reasons.add("letter_digit_token")
        if remaining(_LONG_DIGITS_RE):
            reasons.add("long_digit_run")
    if not ner_used:
        reasons.add("ner_off")
    ordered = [r for r in RISK_WEIGHTS if r in reasons]
    return round(sum(RISK_WEIGHTS[r] for r in ordered), 3), ordered


# ---- batches -------------------------------------------------------------------------------


@dataclass
class RedactionBatch:
    """The result of `redact_batch`: conversations, totals and what tier 2 did."""

    conversations: list[Conversation]
    counts: dict[str, int]
    tier2_mode: str = "off"
    tier2_sent: int = 0  # conversations that went through the tier-2 model
    tier2_risky: int = 0  # conversations at or above the threshold (or all, for "all")
    tier2_turns: int = 0  # turns the model read
    tier2_status: dict[str, Any] | None = None  # pii_model.Status.to_dict() if requested
    risk_reasons: dict[str, int] = field(default_factory=dict)  # signal -> conversations
    seconds: dict[str, float] = field(default_factory=dict)  # stage -> wall time

    def summary(self) -> dict[str, Any]:
        """Everything but the conversations (JSON-able; for progress and reports)."""
        out = asdict(self)
        out.pop("conversations")
        out["conversations"] = len(self.conversations)
        return out


def redact_batch(
    conversations: Sequence[Conversation],
    *,
    profile: Profile = "standard",
    ner: bool = False,
    ner_model: str | None = None,
    surrogates: bool = False,
    tier2: Tier2Mode = "off",
    tier2_threshold: float = DEFAULT_TIER2_THRESHOLD,
    tier2_model: str | None = None,
    tier2_batch_size: int = pii_model.DEFAULT_BATCH_SIZE,
    ner_spans: Sequence[Sequence[Sequence[ner_layer.NerSpan]] | None] | None = None,
    **options: Any,
) -> RedactionBatch:
    """Redact many conversations (one session each) and report what happened.

    1. NER (when `ner=True`) runs over every turn of every conversation in one batch.
    2. Tier 1 (rules + NER spans) runs on every conversation.
    3. `tier2="auto"`: `risk` scores each result; conversations scoring at least
       `tier2_threshold` have all their turns sent through the tier-2 model in one
       batch, the spans are filtered (`pii_model.filter_spans`, which knows the names
       tier 1 registered and which turns answer a name question) and the conversation
       is redacted again with them, with the same salt — so tier 1's placeholders keep
       their numbers. `tier2="all"` sends every conversation. If the model is
       unavailable, the tier-1 results stand.

    Other `options` (pseudonymize_ids, keep_patterns, salt, …) go to the session. With
    no `salt`, each conversation gets its own random one.
    """
    if profile not in PROFILES:
        raise ValueError(f"unknown redaction profile {profile!r} (use one of {PROFILES})")
    convs = list(conversations)
    if profile == "off":
        marked = [_mark_off(c)[0] for c in convs]
        return RedactionBatch(conversations=marked, counts={}, tier2_mode="off")
    started = time.perf_counter()
    seconds: dict[str, float] = {}
    spans_per_conv: list[Sequence[Sequence[ner_layer.NerSpan]] | None]
    if ner_spans is not None:
        spans_per_conv = list(ner_spans)
    else:
        spans_per_conv = [None] * len(convs)
        if ner:
            texts = [t.text for conv in convs for t in conv.turns]
            flat = ner_layer.analyze_texts(texts, profile=profile, model=ner_model)
            pos = 0
            for i, conv in enumerate(convs):
                spans_per_conv[i] = flat[pos : pos + len(conv.turns)]
                pos += len(conv.turns)
        seconds["ner"] = round(time.perf_counter() - started, 3)
    ner_ok = bool(ner_layer.ner_available(ner_model)) if (ner or ner_spans is not None) else False
    shared_salt = options.pop("salt", None)
    salts = [shared_salt or secrets.token_bytes(16) for _ in convs]

    t1 = time.perf_counter()
    results = [
        _redact_one(
            conv,
            profile=profile,
            salt=salt,
            surrogates=surrogates,
            ner_spans=spans,
            ner_used=spans is not None and ner_ok,
            **options,
        )
        for conv, spans, salt in zip(convs, spans_per_conv, salts, strict=True)
    ]
    seconds["tier1"] = round(time.perf_counter() - t1, 3)
    batch = RedactionBatch(conversations=[], counts={}, tier2_mode=tier2)
    if tier2 != "off":
        _run_tier2(
            batch, convs, results, spans_per_conv, salts, ner_ok,
            profile=profile, surrogates=surrogates, tier2=tier2,
            threshold=tier2_threshold, model=tier2_model, batch_size=tier2_batch_size,
            seconds=seconds, options=options,
        )  # fmt: skip
    totals: Counter[str] = Counter()
    for _, counts, _ in results:
        totals.update(counts)
    batch.conversations = [out for out, _, _ in results]
    batch.counts = dict(totals)
    seconds["total"] = round(time.perf_counter() - started, 3)
    batch.seconds = seconds
    return batch


_CAPITALISED_RE = re.compile(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-z]+\b")


def _tier2_wants_turn(before: Turn, after: Turn) -> bool:
    """Should the tier-2 model read this turn of a risky conversation?

    Customer, system and unknown turns: yes (if they have letters) — that is where
    identifiers are introduced. Agent turns mostly echo what the customer said, and an
    echoed name is already replaced once tier 2 registers it from the customer's turn;
    they are read only if something suspicious that was in the original is still there
    (a capitalised word mid-sentence that is not a common word, a letter+digit token,
    an "@", 7+ digits). On ABCD this skips ~40 % of turns.
    """
    if not any(ch.isalpha() for ch in before.text):
        return False
    if before.role != "agent":
        return True
    out, original = after.text, before.text
    for rx in (_LETTER_DIGIT_RE, _AT_RE, _LONG_DIGITS_RE):
        if any(m.group(0) in original for m in rx.finditer(out)):
            return True
    return any(
        m.group(0).lower() not in COMMON_WORDS and m.group(0) in original
        for m in _CAPITALISED_RE.finditer(out)
    )


def _run_tier2(
    batch: RedactionBatch,
    convs: list[Conversation],
    results: list[tuple[Conversation, dict[str, int], RedactionSession]],
    spans_per_conv: list[Sequence[Sequence[ner_layer.NerSpan]] | None],
    salts: list[bytes],
    ner_ok: bool,
    *,
    profile: Profile,
    surrogates: bool,
    tier2: Tier2Mode,
    threshold: float,
    model: str | None,
    batch_size: int,
    seconds: dict[str, float],
    options: dict[str, Any],
) -> None:
    """Triage, run the tier-2 model on the risky conversations, redo those (in place)."""
    reasons_count: Counter[str] = Counter()
    risky: list[int] = []
    for i, (conv, (out, _, _)) in enumerate(zip(convs, results, strict=True)):
        score, reasons = risk(
            conv, out, ner_used=out.meta.get("redaction_ner"),
            keep_patterns=options.get("keep_patterns"), profile=profile,
        )  # fmt: skip
        reasons_count.update(reasons)
        out.meta.update(redaction_risk=score, redaction_risk_reasons=reasons)
        if tier2 == "all" or score >= threshold:
            risky.append(i)
    batch.risk_reasons = dict(reasons_count)
    batch.tier2_risky = len(risky)
    if not risky:
        return
    status = pii_model.pii_model_available(model)
    batch.tier2_status = status.to_dict()
    if not status:
        for i in risky:
            results[i][0].meta["redaction_tier2_unavailable"] = status.reason
        return
    t2 = time.perf_counter()
    # which turns of the risky conversations the model reads (see _tier2_wants_turn)
    wanted = [
        (i, j)
        for i in risky
        for j, (b, a) in enumerate(zip(convs[i].turns, results[i][0].turns, strict=True))
        if _tier2_wants_turn(b, a)
    ]
    found = pii_model.detect(
        [convs[i].turns[j].text for i, j in wanted],
        threshold=pii_model.DEFAULT_THRESHOLD,
        batch_size=batch_size,
        model=model,
    )
    by_turn = dict(zip(wanted, found, strict=True))
    batch.tier2_turns = len(wanted)
    seconds["tier2_model"] = round(time.perf_counter() - t2, 3)
    for i in risky:
        conv, session = convs[i], results[i][2]
        known = set(session._names)
        per_turn: list[list[pii_model.Span]] = []
        for j, t in enumerate(conv.turns):
            prev = _question_before(conv.turns, j)
            per_turn.append(
                pii_model.filter_spans(
                    t.text,
                    by_turn.get((i, j), []),
                    known_names=known,
                    after_name_question=bool(prev and NAME_REQUEST_RE.search(prev))
                    and t.role == "customer",
                    keep=session.keep,
                    common_words=COMMON_WORDS,
                )
            )
        meta_before = results[i][0].meta
        spans = spans_per_conv[i]
        redone = _redact_one(
            conv,
            profile=profile,
            salt=salts[i],
            surrogates=surrogates,
            ner_spans=spans,
            ner_used=spans is not None and ner_ok,
            tier2_spans=per_turn,
            **options,
        )
        redone[0].meta.update(
            redaction_risk=meta_before["redaction_risk"],
            redaction_risk_reasons=meta_before["redaction_risk_reasons"],
        )
        results[i] = redone
    batch.tier2_sent = len(risky)
    seconds["tier2"] = round(time.perf_counter() - t2, 3)


def redact_conversations(
    conversations: list[Conversation],
    *,
    profile: Profile = "standard",
    ner: bool = False,
    ner_model: str | None = None,
    surrogates: bool = False,
    tier2: Tier2Mode = "off",
    tier2_threshold: float = DEFAULT_TIER2_THRESHOLD,
    tier2_model: str | None = None,
    **options: Any,
) -> tuple[list[Conversation], dict[str, int]]:
    """Redact many conversations (one session each). Returns them and total counts.

    With `ner=True` every turn of every conversation goes through the NER model in one
    batch (much faster than per conversation). `tier2="auto"` adds the tier-2 model
    for conversations tier 1 is unsure about (see `redact_batch`, which also reports
    how many were sent; each conversation's `meta["redaction_tier2"]` says too). Other
    options are passed to `redact_conversation`.
    """
    batch = redact_batch(
        conversations,
        profile=profile,
        ner=ner,
        ner_model=ner_model,
        surrogates=surrogates,
        tier2=tier2,
        tier2_threshold=tier2_threshold,
        tier2_model=tier2_model,
        **options,
    )
    return batch.conversations, batch.counts


@dataclass(frozen=True)
class Redactor:
    """A picklable redaction configuration — one object for the corpus ingest and the
    worker, safe to send to a process pool (`corpus.ingest_file_to_store(redact=...)`).

        redact = Redactor(profile="standard", ner=True, surrogates=True, tier2="auto")
        clean, counts = redact(conv)            # one conversation: the Redactor contract
        batch = redact.batch(convs)             # many: NER / tier 2 batched, + stats

    Each process loads NER and the tier-2 model once (they are cached per process);
    with tier 2 on, prefer few worker processes — every one holds its own ~1.2 GB copy.
    """

    profile: Profile = "standard"
    ner: bool = False
    ner_model: str | None = None
    surrogates: bool = False
    tier2: Tier2Mode = "off"
    tier2_threshold: float = DEFAULT_TIER2_THRESHOLD
    tier2_model: str | None = None
    pseudonymize_ids: bool = False
    redact_private_ips: bool = False
    keep_patterns: tuple[str, ...] | None = None
    extra_patterns: tuple[tuple[str, str], ...] = ()

    def _kwargs(self) -> dict[str, Any]:
        return dict(
            profile=self.profile,
            ner=self.ner,
            ner_model=self.ner_model,
            surrogates=self.surrogates,
            tier2=self.tier2,
            tier2_threshold=self.tier2_threshold,
            tier2_model=self.tier2_model,
            pseudonymize_ids=self.pseudonymize_ids,
            redact_private_ips=self.redact_private_ips,
            keep_patterns=self.keep_patterns,
            extra_patterns=dict(self.extra_patterns) or None,
        )

    def batch(self, conversations: Sequence[Conversation]) -> RedactionBatch:
        """Redact many conversations at once (NER and tier 2 batched)."""
        return redact_batch(conversations, **self._kwargs())

    def __call__(self, conv: Conversation) -> tuple[Conversation, dict[str, int]]:
        out = self.batch([conv]).conversations[0]
        return out, dict(out.meta["redaction_counts"])


# ---- JSON exports --------------------------------------------------------------------------

# Values under these keys are a person's name even without a cue ("name": "Priya Shah").
NAME_KEYS: frozenset[str] = frozenset(
    "full_name first_name last_name given_name family_name surname customer_name "
    "patient_name contact_name requester_name requester caller_name agent_name "
    "assignee_name user_name employee_name display_name".split()
)
# A bare "name" key is only a person's name when the record also looks like a person.
_PERSON_SIBLING_KEYS = frozenset(
    "email phone mobile dob date_of_birth birth_date address customer_id patient_id "
    "user_id employee_id contact_id ssn mrn".split()
)


def _name_like(value: str) -> bool:
    words = value.split()
    return 1 <= len(words) <= 4 and not _has_digit(value) and all(w[:1].isupper() for w in words)


def sanitize_json_strings(
    obj: Any,
    profile: Profile = "standard",
    *,
    keep_patterns: KeepPatterns | None = None,
    skip_keys: Sequence[str] = (),
    ner: bool = False,
    ner_model: str | None = None,
    session: RedactionSession | None = None,
    salt: bytes | None = None,
    surrogates: bool = False,
    **options: Any,
) -> tuple[Any, dict[str, int]]:
    """Redact every string value inside a JSON-like structure; return (new_obj, counts).

    Walks dicts, lists and tuples. Keys are kept as they are (they are field names, not
    data); non-string values are kept; strings matching `keep_patterns` are kept; values
    under any key in `skip_keys` (a whole subtree) are copied unchanged — use that for
    fields the environment must keep verbatim (e.g. `skip_keys=("started_on",)` if a
    task's checks depend on exact dates in strict mode).

    One session covers the whole object, so "Priya Shah" in a customer record and in
    the task's goal text become the same `<NAME_1>`. Values under name keys
    (`NAME_KEYS`, or a bare "name" next to email/phone/customer_id/...) are registered
    as names even without a cue. The input is not modified.

    Intended for exits: publishing an environment (`TaskRecord.model_dump()` dicts) and
    exporting training data (chat-format `{"messages": [...]}` examples).

    `surrogates=True` writes consistent fake values instead of placeholders (note that
    text which already carries surrogates gets fresh ones: to this function a fake name
    after "my name is" looks like any other name). With profile "off" this is a no-op:
    the same object comes back, with no counts.
    """
    if session is None and profile == "off":
        return obj, {}
    if session is None:
        session = RedactionSession(
            salt=salt,
            profile=profile,
            keep_patterns=keep_patterns,
            surrogates=surrogates,
            **options,
        )
    if session.profile == "off":
        return obj, {}
    skip = set(skip_keys)

    # pass 1: collect strings (in walk order) and register names from keys and cues
    strings: list[str] = []

    def collect(node: Any) -> None:
        if isinstance(node, dict):
            personal = bool(_PERSON_SIBLING_KEYS & {str(k).lower() for k in node})
            for k, v in node.items():
                if k in skip:
                    continue
                key = str(k).lower()
                if isinstance(v, str) and _name_like(v):
                    if key in NAME_KEYS or (key == "name" and personal):
                        session.add_name(v)
                collect(v)
        elif isinstance(node, list | tuple):
            for v in node:
                collect(v)
        elif isinstance(node, str):
            strings.append(node)

    collect(obj)
    spans = _ner_spans_for(strings, ner=ner, profile=session.profile, ner_model=ner_model)
    for i, s in enumerate(strings):
        if spans is not None:
            session.register_ner(s, spans[i])
        session.collect_names(s)

    # pass 2: rebuild with redacted strings
    def rebuild(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: (v if k in skip else rebuild(v)) for k, v in node.items()}
        if isinstance(node, list):
            return [rebuild(v) for v in node]
        if isinstance(node, tuple):
            return tuple(rebuild(v) for v in node)
        if isinstance(node, str):
            if not node or session._is_kept(node):
                return node
            return session.redact(node).text
        return node

    return rebuild(obj), dict(session.counts)


# ---- auditing ------------------------------------------------------------------------------


@dataclass
class Finding:
    """Something in (already redacted) text that still looks like personal data."""

    kind: str
    text: str
    start: int
    end: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Plain dict (for reports)."""
        return asdict(self)


_NAME_BIGRAM_RE = re.compile(rf"\b({_NAME_WORD_X})\s+({_NAME_WORD_X})(?![\w'’-])")
_PHONE_LIKE_RE = re.compile(r"(?<![\w.])\+?\(?\d[\d ().-]{5,20}\d(?![\w.])")


def scan_text(
    text: str, profile: Profile = "standard", *, keep_patterns: KeepPatterns | None = None
) -> list[Finding]:
    """Detector-only pass: what in `text` still looks like PII? Changes nothing.

    Meant for text that has already been redacted (and for re-scanning exports). It is
    deliberately more suspicious than the redactor: capitalised name-like bigrams whose
    words are not in `COMMON_WORDS`, names after cues or titles, digit runs (5+ digits
    in strict, 7+ in standard), phone-like numbers, emails, street addresses, and in
    strict also URLs, any IP, dates, ages over 89, VINs and ZIPs. Placeholders and
    values matching `keep_patterns` are ignored. Expect false positives — the point is
    a short list for a human to look at.
    """
    if profile not in PROFILES:
        raise ValueError(f"unknown redaction profile {profile!r}")
    if profile == "off":  # nothing was redacted: look with the standard detectors
        profile = "standard"
    strict = profile == "strict"
    keep = _compile_keep(keep_patterns, profile)
    probe = RedactionSession(profile=profile, keep_patterns=keep, salt=b"scan")
    findings: list[Finding] = []

    def add(kind: str, start: int, end: int, reason: str) -> None:
        findings.append(Finding(kind, text[start:end], start, end, reason))

    def check_all(rx: re.Pattern[str], kind: str, reason: str, ok: Callable | None = None):
        for start, end, m in _finditer_outside(rx, text, keep):
            if "value" in m.re.groupindex and m.group("value") is not None:
                start, end = (
                    start + m.start("value") - m.start(),
                    start + m.end("value") - m.start(),
                )
            if ok is None or ok(text[start:end]):
                add(kind, start, end, reason)

    check_all(EMAIL_RE, "EMAIL", "email address")
    check_all(URL_RE, "URL", "URL" if strict else "URL with a secret", probe._url_check)
    check_all(SSN_RE, "SSN", "SSN-shaped number")
    check_all(CARD_RE, "CARD", "Luhn-valid card number", probe._card_check)
    check_all(IPV4_RE, "IP", "IP address", probe._ip_check)
    check_all(IPV6_RE, "IP", "IPv6 address", probe._ip_check)
    check_all(MAC_RE, "MAC", "MAC address")
    check_all(
        _PHONE_LIKE_RE,
        "PHONE",
        "phone-like number",
        lambda v: (
            7 <= len(re.sub(r"\D", "", v)) <= 15
            and not _DATE_LIKE.fullmatch(v)
            and bool(re.search(r"[\s().-]", v) or v.startswith("+"))
        ),
    )
    check_all(ADDRESS_RE, "ADDRESS", "street address")
    check_all(DOB_RE, "DOB", "labelled date of birth")
    for name in probe.find_names(text):
        for start, end, _ in _finditer_outside(re.compile(rf"\b{re.escape(name)}\b"), text, keep):
            add("NAME", start, end, "name after a cue")
    check_all(NAME_TITLE_RE, "NAME", "name after a title", lambda v: _plausible_name(v) is not None)
    check_all(
        NAME_LOOSE_GREETING_RE,
        "NAME",
        "name after a greeting",
        lambda v: _plausible_name(v) is not None,
    )
    for start, end, m in _finditer_outside(_NAME_BIGRAM_RE, text, keep):
        if m.group(1).lower() not in COMMON_WORDS and m.group(2).lower() not in COMMON_WORDS:
            add("NAME", start, end, "capitalised bigram not in the common-word list")
    digits = r"\d{5,}" if strict else r"\d{7,}"
    check_all(re.compile(rf"(?<![\w.]){digits}(?![\w.])"), "DIGITS", "long digit run")
    if strict:
        check_all(STRICT_DATE_RE, "DATE", "date (month/day)", _date_ok)
        check_all(AGE_SUFFIX_RE, "AGE", "age over 89")
        check_all(AGE_CUE_RE, "AGE", "age over 89")
        check_all(VIN_RE, "VEHICLE", "VIN-shaped token", probe._vin_check)
        check_all(ZIP_STATE_RE, "ZIP", "ZIP code after a state")
        check_all(LONG_ID_RE, "ID", "token with 6+ digits", probe._long_id_check)
        for kind, rx in STRICT_CUE_PATTERNS:
            check_all(rx, kind, f"{kind.lower()} after a cue", _id_value_ok)

    # drop findings contained in an earlier (longer) one, then sort by position
    findings.sort(key=lambda f: (f.start, -(f.end - f.start)))
    kept: list[Finding] = []
    for f in findings:
        if kept and f.start >= kept[-1].start and f.end <= kept[-1].end:
            continue
        kept.append(f)
    return kept


def _snippet(text: str, start: int, end: int, width: int) -> str:
    """`text[start:end]` with up to `width` characters of context on each side."""
    a, b = max(0, start - width), min(len(text), end + width)
    return ("…" if a else "") + text[a:b] + ("…" if b < len(text) else "")


def build_redaction_report(
    redacted: Sequence[Conversation],
    *,
    profile: Profile = "standard",
    counts: dict[str, int] | None = None,
    ner_requested: bool = False,
    ner_model: str | None = None,
    keep_patterns: KeepPatterns | None = None,
    n_input: int | None = None,
    sample_size: int = 5,
    snippet_chars: int = 240,
    max_remnants: int = 100,
    seed: int = 0,
) -> dict[str, Any]:
    """A JSON-able summary of a redaction run, for a human sign-off step.

    Takes REDACTED conversations only and never sees the originals, so the report
    cannot leak them. It contains: the profile, counts by type (from `counts` or summed
    from each conversation's `meta["redaction_counts"]`), NER status, the keep
    patterns, a deterministic sample of redacted snippets (turns that contain
    placeholders, so a reviewer sees redaction at work), and "suspicious remnants" from
    `scan_text`, each with a little context.

    Remnants are excerpts of the redacted text that may themselves be missed personal
    data — treat the report with the same care as the redacted dataset.
    """
    if counts is None:
        total: Counter[str] = Counter()
        for conv in redacted:
            total.update(conv.meta.get("redaction_counts") or {})
        counts = dict(total)
    status = ner_layer.ner_available(ner_model) if ner_requested else None
    keep = _compile_keep(keep_patterns, profile)

    with_placeholders = [
        (conv.id, t.role, t.text)
        for conv in redacted
        for t in conv.turns
        if PLACEHOLDER_RE.search(t.text)
    ]
    rng = random.Random(seed)
    picked = rng.sample(with_placeholders, min(sample_size, len(with_placeholders)))
    samples = [
        {"conversation": cid, "role": role, "text": text[:snippet_chars]}
        for cid, role, text in picked
    ]

    remnants: list[dict[str, Any]] = []
    remnant_counts: Counter[str] = Counter()
    for conv in redacted:
        for i, t in enumerate(conv.turns):
            for f in scan_text(t.text, profile, keep_patterns=keep):
                remnant_counts[f.kind] += 1
                if len(remnants) < max_remnants:
                    remnants.append(
                        {
                            "conversation": conv.id,
                            "turn": i,
                            "kind": f.kind,
                            "text": f.text,
                            "reason": f.reason,
                            "context": _snippet(t.text, f.start, f.end, 40),
                        }
                    )
    return {
        "profile": profile,
        "conversations": len(redacted),
        "input_conversations": n_input if n_input is not None else len(redacted),
        "counts_by_type": dict(sorted(counts.items())),
        "total_redactions": sum(counts.values()),
        "ner": {
            "requested": ner_requested,
            **(status.to_dict() if status else {"available": False, "model": None, "reason": ""}),
        },
        "keep_patterns": [rx.pattern for rx in keep],
        "samples": samples,
        "remnant_counts": dict(sorted(remnant_counts.items())),
        "remnants": remnants,
        "remnants_truncated": sum(remnant_counts.values()) > len(remnants),
        "note": (
            "Automated, best-effort redaction — a technical safeguard, not a HIPAA or GDPR "
            "certification. Review the samples and remnants before sharing."
        ),
    }


def audit(
    conversations: list[Conversation],
    *,
    profile: Profile = "standard",
    ner: bool = False,
    ner_model: str | None = None,
    keep_patterns: KeepPatterns | None = None,
    sample_size: int = 5,
    **options: Any,
) -> tuple[list[Conversation], dict[str, Any]]:
    """Redact `conversations` and build the report in one go: (redacted, report).

    The report is built from the redacted output only (see `build_redaction_report`).
    """
    redacted, counts = redact_conversations(
        conversations,
        profile=profile,
        ner=ner,
        ner_model=ner_model,
        keep_patterns=keep_patterns,
        **options,
    )
    report = build_redaction_report(
        redacted,
        profile=profile,
        counts=counts,
        ner_requested=ner,
        ner_model=ner_model,
        keep_patterns=keep_patterns,
        n_input=len(conversations),
        sample_size=sample_size,
    )
    return redacted, report
