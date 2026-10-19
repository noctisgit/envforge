"""A small, deterministic SIP/VoIP call-routing simulator.

No packets are sent. The "voice platform" is data: two PBXes with dial plans,
an SBC with carrier trunks and an inbound DID table, DNS records, trunk
credentials, phones and their registrations. `call()` walks one call through
that data the way a real SIP proxy chain would, and records a SIP ladder
(INVITE -> 100 Trying -> ... -> 200 OK, or the error response) so the agent can
see *where* and *why* a call fails.

It is small on purpose. The point is not to be a SIP stack; it is a world with
enough structure that (a) faults have distinct, discoverable symptoms
(404 / 403 / 408 / 480 / 484 / 488 / 503), (b) fixes can break other calls,
and (c) success is checkable by code.

Topology (fixed; what varies per task is phones, called numbers, faults and decoys):

    phones --- pbx-hq (HQ, ext 1xx) ---+                   +--- carrier-a  national PSTN; hosts our DIDs
                  ||                   +----- sbc1 --------+--- carrier-b  international / backup
              intersite trunk          |                   +--- e911       emergency services
                  ||                   |
    phones --- pbx-br (branch, 2xx) ---+

Outbound: phone -> its PBX dial plan -> (local ext | intersite trunk | sbc1 -> carrier trunk).
Inbound:  PSTN -> carrier-a -> sbc1 (peer check, DID table) -> PBX -> phone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

SIP_PORT = 5060
SBC = "sbc1"
PBXES = ("pbx-hq", "pbx-br")
TABLES = (*PBXES, SBC)  # dial plans per PBX + the SBC's inbound DID table
KNOWN_CODECS = ("G722", "PCMU", "PCMA", "G729", "OPUS")
PSTN_CODECS = ["PCMU", "PCMA", "G729"]  # what an outside caller's carrier offers
E164 = re.compile(r"\+[1-9]\d{6,14}")

REASONS = {
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not Found",
    408: "Request Timeout",
    480: "Temporarily Unavailable",
    484: "Address Incomplete",
    488: "Not Acceptable Here",
    503: "Service Unavailable",
}


def status(code: int) -> str:
    """SIP status line text, e.g. 488 -> '488 Not Acceptable Here'."""
    return f"{code} {REASONS.get(code, 'Error')}"


def sdp(codecs: list[str]) -> str:
    return f"  [SDP: {','.join(codecs)}]"


@dataclass
class Rule:
    """One dial-plan rule (or SBC inbound DID rule).

    Rules are tried lowest `priority` first; the first whose `pattern` fully
    matches the dialled number wins. A `route` rule rewrites the number
    (drop `strip` leading characters, then add `prepend`) and sends it to
    `target`; a `reject` rule answers with SIP `code`.
    """

    id: int
    priority: int
    pattern: str  # regular expression, matched against the whole number
    action: str = "route"  # "route" | "reject"
    target: str = ""  # route: "local", a trunk name, or (SBC table) a PBX name
    strip: int = 0
    prepend: str = ""
    code: int = 403  # reject only
    note: str = ""

    def matches(self, number: str) -> bool:
        try:
            return re.fullmatch(self.pattern, number) is not None
        except re.error:
            return False

    def translate(self, number: str) -> str:
        return self.prepend + number[self.strip :]

    def signature(self) -> tuple:
        """Everything that defines what the rule *does*, except where it sits in the order."""
        return (self.pattern, self.action, self.target, self.strip, self.prepend, self.code)

    def render(self) -> str:
        if self.action == "reject":
            act, xlate, target = f"reject {self.code}", "-", "-"
        else:
            parts = []
            if self.strip:
                parts.append(f"strip {self.strip}")
            if self.prepend:
                parts.append(f"prepend {self.prepend}")
            act, xlate, target = "route", ", ".join(parts) or "-", self.target
        note = f"   ! {self.note}" if self.note else ""
        return (
            f"{self.id:>4} {self.priority:>5}  {self.pattern:<22} {act:<11} "
            f"{xlate:<20} {target:<10}{note}"
        ).rstrip()


@dataclass
class Trunk:
    """A SIP trunk. Carrier trunks live on the SBC; the intersite trunk joins the two PBXes."""

    name: str
    kind: str  # "carrier" | "intersite"
    target: str  # carrier: SIP domain (SRV) or host; intersite: description only
    codecs: list[str]
    max_channels: int
    active_calls: int
    enabled: bool = True
    auth_ref: str = ""  # credential reference; "" = IP-authenticated
    note: str = ""


@dataclass
class Carrier:
    """The far end of a carrier trunk. The agent cannot configure this; it is the outside world."""

    name: str  # label in the SIP ladder
    domain: str
    ips: list[str]  # addresses the carrier really listens on (and sends from)
    codecs: list[str]
    realm: str = ""  # "" = no digest challenge
    account: str = ""  # username the carrier expects
    national_only: bool = False  # rejects non +1 numbers
    emergency_only: bool = False  # only terminates "911"


@dataclass
class Credential:
    ref: str
    realm: str
    username: str
    status: str = "active"  # "active" | "rotated"


@dataclass
class Extension:
    number: str
    name: str
    pbx: str
    model: str
    ip: str  # the phone's current address (DHCP lease)
    codecs: list[str]
    did: str = ""  # E.164 direct-dial number, "" if none


@dataclass
class Registration:
    contact: str  # IP the PBX will send INVITEs to
    expires: int  # seconds left; <= 0 means expired


@dataclass
class CallResult:
    ok: bool
    ladder: list[str]
    reason: str  # human-readable failure reason, "" when ok
    egress: str = ""  # trunk the call left through ("" if internal or rejected)
    codec: str = ""
    path: list[str] = field(default_factory=list)  # nodes the INVITE reached, caller first
    blame: str = ""  # failed calls: the element at fault (a PBX, sbc1, trunk or extension)


class Ladder:
    """Builds a SIP ladder diagram while a call is walked through the platform.

    `path` is the chain of nodes the INVITE has crossed so far; final responses
    walk it backwards, hop by hop, as they would through real proxies. `router` is
    the node whose rule last chose the destination number: when the far end says
    "no such number", that is where the bad translation happened.
    """

    def __init__(self, caller: str) -> None:
        self.lines: list[str] = []
        self.path: list[str] = [caller]
        self.egress = ""
        self.router = ""

    def send(self, src: str, dst: str, msg: str) -> None:
        self.lines.append(f"  {src:<14} -> {dst:<14} {msg}")

    def note(self, node: str, text: str) -> None:
        self.lines.append(f"  {'':<14}    [{node}] {text}")

    def invite(self, dst: str, uri: str, extra: str = "", trying: bool = True) -> None:
        here = self.path[-1]
        self.send(here, dst, f"INVITE {uri}{extra}")
        if trying:
            self.send(dst, here, "100 Trying")
        self.path.append(dst)

    def fail(
        self, code: int, detail: str = "", origin: str | None = None, blame: str = ""
    ) -> CallResult:
        """Send final response `code` from `origin` (default: the last node) back to the caller.

        `blame` names the element at fault when it is not the node that answered
        (a trunk, a phone, or the node that translated the number); it only feeds
        the diagram, never the ladder.
        """
        nodes = self.path if origin is None else self.path[: self.path.index(origin) + 1]
        for i in range(len(nodes) - 1, 0, -1):
            self.send(nodes[i], nodes[i - 1], status(code))
        why = f"{status(code)} from {nodes[-1]}" + (f" ({detail})" if detail else "")
        return CallResult(
            False, self.lines, why, self.egress, path=list(self.path), blame=blame or nodes[-1]
        )

    def answer(self, codec: str) -> CallResult:
        """The last node answers: 180 and 200 travel back, the caller ACKs end to end."""
        for msg in ("180 Ringing", f"200 OK  [SDP: {codec}]"):
            for i in range(len(self.path) - 1, 0, -1):
                self.send(self.path[i], self.path[i - 1], msg)
        self.send(self.path[0], self.path[-1], "ACK")
        return CallResult(True, self.lines, "", self.egress, codec, path=list(self.path))


class Platform:
    """The whole voice platform: config the agent can change plus the outside world it can't."""

    def __init__(self) -> None:
        self.tables: dict[str, list[Rule]] = {t: [] for t in TABLES}
        self.trunks: dict[str, Trunk] = {}
        self.carriers: dict[str, Carrier] = {}  # keyed by trunk name
        self.dns_a: dict[str, str] = {}
        self.dns_srv: dict[str, tuple[str, int]] = {}  # "_sip._udp.<domain>" -> (host, port)
        self.credentials: dict[str, Credential] = {}
        self.extensions: dict[str, Extension] = {}
        self.registrations: dict[str, Registration] = {}
        self.pstn: dict[str, str] = {}  # outside numbers that answer -> who they are
        self.pstn_ranges: list[str] = []  # regexes of number ranges that always answer
        self.did_carrier = "carrier-a"  # trunk whose carrier delivers our inbound calls
        self.did_block = r"\+14155550\d{3}"  # DIDs that carrier routes to us

    # ---- construction ----------------------------------------------------
    def add_extension(self, ext: Extension, expires: int = 3600) -> None:
        self.extensions[ext.number] = ext
        self.registrations[ext.number] = Registration(ext.ip, expires)

    def add_trunk(self, trunk: Trunk, carrier: Carrier | None = None) -> None:
        self.trunks[trunk.name] = trunk
        if carrier is not None:
            self.carriers[trunk.name] = carrier

    # ---- lookups ---------------------------------------------------------
    def rules(self, table: str) -> list[Rule]:
        """A table's rules in evaluation order."""
        return sorted(self.tables[table], key=lambda r: (r.priority, r.id))

    def rule(self, table: str, rule_id: int) -> Rule | None:
        return next((r for r in self.tables[table] if r.id == rule_id), None)

    def match(self, table: str, number: str) -> Rule | None:
        """First matching rule, or None."""
        return next((r for r in self.rules(table) if r.matches(number)), None)

    def resolve(self, target: str) -> tuple[str | None, int, list[str], str]:
        """Resolve a trunk target the way an SBC does: SRV first, then A.

        Returns (ip, port, notes for the ladder, error). `ip` is None on failure.
        """
        notes: list[str] = []
        host, port = target, SIP_PORT
        srv_name = f"_sip._udp.{target}"
        if srv_name in self.dns_srv:
            host, port = self.dns_srv[srv_name]
            notes.append(f"DNS SRV {srv_name} -> {host}:{port}")
        ip = self.dns_a.get(host)
        if ip is None:
            return None, port, notes, f"DNS A {host}: NXDOMAIN"
        notes.append(f"DNS A {host} -> {ip}")
        return ip, port, notes, ""

    def pstn_answers(self, number: str) -> bool:
        return number in self.pstn or any(re.fullmatch(p, number) for p in self.pstn_ranges)

    def peer_trunk(self, ip: str) -> Trunk | None:
        """Which carrier trunk an inbound INVITE from `ip` belongs to (by resolved target)."""
        for t in sorted(self.trunks.values(), key=lambda t: t.name):
            if t.kind == "carrier" and self.resolve(t.target)[0] == ip:
                return t
        return None

    # ---- call processing -------------------------------------------------
    def call(self, src: str, dialed: str) -> CallResult:
        """Place one call. `src` is an extension number or an outside E.164 number."""
        if src in self.extensions:
            return self._from_extension(self.extensions[src], dialed)
        if E164.fullmatch(src):
            return self._from_pstn(src, dialed)
        return CallResult(False, [], f"unknown caller '{src}'")

    def _from_extension(self, ext: Extension, dialed: str) -> CallResult:
        lad = Ladder(ext.number)
        pbx = ext.pbx
        lad.router = pbx
        lad.invite(pbx, f"sip:{dialed}@{pbx}", sdp(ext.codecs))
        rule = self.match(pbx, dialed)
        if rule is None:
            lad.note(pbx, f"no dial-plan rule matches '{dialed}'")
            return lad.fail(404, f"no dial-plan match on {pbx}")
        matched = f"dial-plan rule {rule.id} (prio {rule.priority}, {rule.pattern}) matched"
        if rule.action == "reject":
            lad.note(
                pbx, f"{matched} -> reject {rule.code}" + (f"  ! {rule.note}" if rule.note else "")
            )
            return lad.fail(rule.code, f"rejected by {pbx} rule {rule.id}")
        number = rule.translate(dialed)
        xlate = f"; {dialed} -> {number}" if number != dialed else ""
        lad.note(pbx, f"{matched}{xlate}; target {rule.target}")
        return self._route(lad, pbx, rule.target, number, ext.codecs)

    def _route(
        self, lad: Ladder, pbx: str, target: str, number: str, offer: list[str]
    ) -> CallResult:
        if target == "local":
            return self._ring(lad, pbx, number, offer)
        trunk = self.trunks.get(target)
        if trunk is None:
            lad.note(pbx, f"route target '{target}' is not a known trunk")
            return lad.fail(503, f"unknown route target {target}")
        if trunk.kind == "intersite":
            return self._via_intersite(lad, pbx, trunk, number, offer)
        return self._via_carrier(lad, trunk, number, offer)

    def _trunk_checks(
        self, lad: Ladder, node: str, t: Trunk, offer: list[str]
    ) -> CallResult | list[str]:
        """State, capacity and codec policy. Returns the filtered offer, or a failed CallResult."""
        if not t.enabled:
            lad.note(node, f"trunk {t.name} is administratively disabled")
            return lad.fail(503, f"trunk {t.name} disabled", blame=t.name)
        if t.active_calls >= t.max_channels:
            lad.note(node, f"trunk {t.name} at channel limit ({t.active_calls}/{t.max_channels})")
            return lad.fail(503, f"trunk {t.name} has no free channels", blame=t.name)
        allowed = [c for c in offer if c in t.codecs]
        if not allowed:
            lad.note(
                node,
                f"no codec in offer [{','.join(offer)}] is allowed on trunk {t.name} "
                f"[{','.join(t.codecs)}]",
            )
            return lad.fail(488, f"codec policy on trunk {t.name}", blame=t.name)
        return allowed

    def _via_intersite(
        self, lad: Ladder, pbx: str, t: Trunk, number: str, offer: list[str]
    ) -> CallResult:
        checked = self._trunk_checks(lad, pbx, t, offer)
        if isinstance(checked, CallResult):
            return checked
        peer = PBXES[1] if pbx == PBXES[0] else PBXES[0]
        lad.egress = t.name
        lad.invite(peer, f"sip:{number}@{peer}", sdp(checked))
        return self._ring(lad, peer, number, checked)

    def _via_carrier(self, lad: Ladder, t: Trunk, number: str, offer: list[str]) -> CallResult:
        lad.invite(SBC, f"sip:{number}@{SBC}", sdp(offer))
        checked = self._trunk_checks(lad, SBC, t, offer)
        if isinstance(checked, CallResult):
            return checked
        ip, port, notes, err = self.resolve(t.target)
        for n in notes:
            lad.note(SBC, n)
        if ip is None:
            lad.note(SBC, f"cannot resolve trunk {t.name} target: {err}")
            return lad.fail(503, err, blame=t.name)
        lad.egress = t.name
        carrier = self.carriers[t.name]
        if ip not in carrier.ips or port != SIP_PORT:
            lad.send(SBC, ip, f"INVITE sip:{number}@{ip}:{port}{sdp(checked)}")
            lad.note(SBC, f"no response from {ip}:{port}; INVITE retransmitted 7x, Timer B fired")
            return lad.fail(408, f"no response from {ip}:{port}", blame=t.name)
        lad.invite(carrier.name, f"sip:{number}@{ip}:{port}", sdp(checked))
        if carrier.realm:
            lad.send(carrier.name, SBC, f'401 Unauthorized  (Digest realm="{carrier.realm}")')
            lad.send(SBC, carrier.name, "ACK")
            cred = self.credentials.get(t.auth_ref)
            if cred is None:
                lad.note(SBC, f"trunk {t.name} has no credential for realm {carrier.realm}")
                return lad.fail(401, "no credentials", origin=SBC, blame=t.name)
            lad.send(
                SBC,
                carrier.name,
                f'INVITE sip:{number}@{ip}:{port}  (Authorization: Digest username="{cred.username}", '
                f'realm="{cred.realm}")',
            )
            if (cred.realm, cred.username, cred.status) != (
                carrier.realm,
                carrier.account,
                "active",
            ):
                return lad.fail(
                    403, f"{carrier.name} rejected credentials '{cred.ref}'", blame=t.name
                )
            lad.send(carrier.name, SBC, "100 Trying")
        return self._terminate(lad, t, carrier, number, checked)

    def _terminate(
        self, lad: Ladder, t: Trunk, carrier: Carrier, number: str, offer: list[str]
    ) -> CallResult:
        """The carrier delivers the call to the outside party (or refuses).

        A refused *number* is blamed on the dial plan that produced it (`lad.router`).
        """
        common = [c for c in offer if c in carrier.codecs]
        if not common:
            return lad.fail(
                488, f"{carrier.name} supports only {','.join(carrier.codecs)}", blame=t.name
            )
        if carrier.emergency_only:
            if number != "911":
                return lad.fail(404, f"{carrier.name} only terminates 911", blame=lad.router)
            lad.invite("PSAP", "sip:911@psap", trying=False)
            return lad.answer(common[0])
        if not E164.fullmatch(number):
            return lad.fail(484, f"'{number}' is not an E.164 number", blame=lad.router)
        if carrier.national_only and not number.startswith("+1"):
            return lad.fail(
                403,
                f"international calling not enabled on {carrier.name} account",
                blame=lad.router,
            )
        if not self.pstn_answers(number):
            return lad.fail(404, f"{number} is not in service", blame=lad.router)
        lad.invite(number, f"tel:{number}", trying=False)
        return lad.answer(common[0])

    def _ring(self, lad: Ladder, pbx: str, number: str, offer: list[str]) -> CallResult:
        """Deliver to a phone registered on `pbx`."""
        ext = self.extensions.get(number)
        if ext is None or ext.pbx != pbx:
            lad.note(pbx, f"no extension {number} on {pbx}")
            return lad.fail(404, f"no extension {number} on {pbx}", blame=lad.router or pbx)
        reg = self.registrations.get(number)
        if reg is None or reg.expires <= 0:
            lad.note(pbx, f"extension {number} has no active registration")
            return lad.fail(480, f"{number} not registered", blame=number)
        lad.invite(number, f"sip:{number}@{reg.contact}:{SIP_PORT}", sdp(offer), trying=False)
        if reg.contact != ext.ip:
            lad.note(pbx, f"no response from {reg.contact}; INVITE retransmitted 7x, Timer B fired")
            return lad.fail(
                408, f"no response from {number} contact {reg.contact}", origin=pbx, blame=number
            )
        common = [c for c in offer if c in ext.codecs]
        if not common:
            # the trunk the call came over narrowed the offer to codecs the phone lacks
            via = self.trunks.get(lad.egress)
            narrowed = via is not None and not set(via.codecs) & set(ext.codecs)
            return lad.fail(
                488,
                f"phone {number} supports only {','.join(ext.codecs)}",
                blame=via.name if via is not None and narrowed else number,
            )
        return lad.answer(common[0])

    def _from_pstn(self, caller: str, called: str) -> CallResult:
        carrier = self.carriers[self.did_carrier]
        lad = Ladder(caller)
        lad.invite(carrier.name, f"sip:{called}@{carrier.domain}", sdp(PSTN_CODECS))
        if not re.fullmatch(self.did_block, called):
            return lad.fail(404, f"{called} is not a DID routed to this customer")
        src_ip = carrier.ips[0]
        lad.invite(SBC, f"sip:{called}@{SBC}  (from {src_ip})", sdp(PSTN_CODECS))
        trunk = self.peer_trunk(src_ip)
        if trunk is None:
            lad.note(SBC, f"INVITE from {src_ip} does not match any carrier trunk peer")
            return lad.fail(403, f"unknown peer {src_ip}", blame=self.did_carrier)
        lad.egress = trunk.name
        checked = self._trunk_checks(lad, SBC, trunk, PSTN_CODECS)
        if isinstance(checked, CallResult):
            return checked
        rule = self.match(SBC, called)
        if rule is None:
            lad.note(SBC, f"no inbound rule matches '{called}'")
            return lad.fail(404, f"no inbound DID match on {SBC}")
        matched = f"inbound rule {rule.id} (prio {rule.priority}, {rule.pattern}) matched"
        if rule.action == "reject":
            lad.note(
                SBC, f"{matched} -> reject {rule.code}" + (f"  ! {rule.note}" if rule.note else "")
            )
            return lad.fail(rule.code, f"rejected by {SBC} rule {rule.id}")
        number = rule.translate(called)
        lad.router = SBC
        lad.note(SBC, f"{matched}; {called} -> {number}; target {rule.target}")
        if rule.target not in PBXES:
            return lad.fail(503, f"inbound target '{rule.target}' is not a PBX")
        lad.invite(rule.target, f"sip:{number}@{rule.target}", sdp(checked))
        return self._ring(lad, rule.target, number, checked)

    # ---- rendering -------------------------------------------------------
    def render_dialplan(self, table: str) -> str:
        title = (
            f"{SBC} inbound DID routing"
            if table == SBC
            else f"{table} dial plan (number as dialled)"
        )
        lines = [
            f"{title}; lowest priority first, first match wins, patterns are full-match regex",
            f"{'id':>4} {'prio':>5}  {'pattern':<22} {'action':<11} {'translate':<20} target",
        ]
        lines += [r.render() for r in self.rules(table)]
        return "\n".join(lines)

    def render_trunks(self) -> str:
        lines = [
            "trunks",
            f"  {'name':<10} {'type':<10} {'state':<9} {'target':<22} {'auth':<20} "
            f"{'codecs':<16} channels",
        ]
        for t in sorted(self.trunks.values(), key=lambda t: t.name):
            target = t.target if t.kind == "carrier" else "pbx-hq <-> pbx-br"
            auth = t.auth_ref or ("ip-auth" if t.kind == "carrier" else "-")
            note = f"   ! {t.note}" if t.note else ""
            lines.append(
                f"  {t.name:<10} {t.kind:<10} {'enabled' if t.enabled else 'DISABLED':<9} "
                f"{target:<22} {auth:<20} {','.join(t.codecs):<16} "
                f"{t.active_calls}/{t.max_channels}{note}"
            )
        return "\n".join(lines)

    def render_dns(self) -> str:
        lines = ["DNS records (as seen by sbc1)"]
        for name, (host, port) in sorted(self.dns_srv.items()):
            lines.append(f"  {name:<28} SRV  10 50 {port} {host}")
        for name, ip in sorted(self.dns_a.items()):
            lines.append(f"  {name:<28} A    {ip}")
        return "\n".join(lines)

    def render_credentials(self) -> str:
        lines = ["trunk credentials (secrets not shown)"]
        for c in sorted(self.credentials.values(), key=lambda c: c.ref):
            lines.append(f"  {c.ref:<22} realm={c.realm:<16} user={c.username:<12} {c.status}")
        return "\n".join(lines)

    def render_extensions(self, pbx: str) -> str:
        lines = [
            f"{pbx} extensions",
            f"  {'ext':<5} {'name':<14} {'DID':<14} {'phone':<14} {'phone IP':<12} codecs",
        ]
        for e in sorted(self.extensions.values(), key=lambda e: e.number):
            if e.pbx == pbx:
                lines.append(
                    f"  {e.number:<5} {e.name:<14} {e.did or '-':<14} {e.model:<14} {e.ip:<12} "
                    f"{','.join(e.codecs)}"
                )
        return "\n".join(lines)

    def render_registrations(self, pbx: str) -> str:
        lines = [f"{pbx} registrations", f"  {'ext':<5} {'contact':<30} expires"]
        for e in sorted(self.extensions.values(), key=lambda e: e.number):
            if e.pbx != pbx:
                continue
            reg = self.registrations.get(e.number)
            if reg is None:
                lines.append(f"  {e.number:<5} {'-':<30} (none)")
                continue
            exp = f"{reg.expires}s" if reg.expires > 0 else "EXPIRED"
            lines.append(f"  {e.number:<5} {f'sip:{e.number}@{reg.contact}:{SIP_PORT}':<30} {exp}")
        return "\n".join(lines)

    def snapshot(self) -> dict:
        return {
            "tables": {t: [r.render() for r in self.rules(t)] for t in TABLES},
            "trunks": {
                t.name: {
                    "enabled": t.enabled,
                    "auth_ref": t.auth_ref,
                    "codecs": list(t.codecs),
                    "max_channels": t.max_channels,
                }
                for t in self.trunks.values()
            },
            "dns_a": dict(self.dns_a),
            "dns_srv": {k: f"{h}:{p}" for k, (h, p) in self.dns_srv.items()},
            "registrations": {
                k: f"{r.contact} ({r.expires}s)" for k, r in self.registrations.items()
            },
        }


# ---- the base platform ------------------------------------------------------

POLICY_E911 = "POLICY E911: emergency calls must use the emergency trunk"
POLICY_PREMIUM = "POLICY COS-001: premium-rate numbers blocked; do not remove"


def _pbx_dialplan(local: str, remote: str) -> list[Rule]:
    """Dial plan shared by both PBXes; `local` / `remote` are the ext patterns (1xx / 2xx)."""
    return [
        Rule(1, 10, r"911", target="emergency", note=POLICY_E911),
        Rule(2, 11, r"9911", target="emergency", strip=1, note=POLICY_E911),
        Rule(3, 20, r"91900\d{7}", action="reject", code=403, note=POLICY_PREMIUM),
        Rule(4, 21, r"91976\d{7}", action="reject", code=403, note=POLICY_PREMIUM),
        Rule(5, 30, local, target="local"),
        Rule(6, 40, remote, target="intersite"),
        Rule(7, 50, r"91[2-9]\d{9}", target="carrier-a", strip=1, prepend="+"),
        Rule(8, 60, r"9011[1-9]\d{6,13}", target="carrier-b", strip=4, prepend="+"),
        Rule(9, 99, r".*", action="reject", code=404, note="intercept: number not in service"),
    ]


def build_base_platform() -> Platform:
    """A fully working platform with the fixed infrastructure and the two special phones.

    Faults, decoys, ordinary phones and outside numbers are added on top of this.
    """
    p = Platform()
    p.tables["pbx-hq"] = _pbx_dialplan(r"1\d\d", r"2\d\d")
    p.tables["pbx-br"] = _pbx_dialplan(r"2\d\d", r"1\d\d")
    p.tables[SBC] = [
        Rule(1, 10, r"\+141555501\d\d", target="pbx-hq", strip=9),
        Rule(2, 20, r"\+141555502\d\d", target="pbx-br", strip=9),
        Rule(3, 99, r".*", action="reject", code=404, note="unassigned DID"),
    ]

    p.add_trunk(
        Trunk(
            "carrier-a",
            "carrier",
            "carrier-a.net",
            ["PCMU", "PCMA"],
            30,
            7,
            auth_ref="cred-carrier-a",
        ),
        Carrier(
            "carrier-a",
            "carrier-a.net",
            ["198.51.100.10"],
            ["PCMU", "PCMA", "G729"],
            realm="carrier-a.net",
            account="acme-voice",
            national_only=True,
        ),
    )
    p.add_trunk(
        Trunk(
            "carrier-b",
            "carrier",
            "carrier-b.net",
            ["PCMA", "PCMU"],
            20,
            1,
            auth_ref="cred-carrier-b",
        ),
        Carrier(
            "carrier-b",
            "carrier-b.net",
            ["203.0.113.20"],
            ["PCMA", "PCMU", "G729"],
            realm="carrier-b.net",
            account="acme-intl",
        ),
    )
    p.add_trunk(
        Trunk(
            "emergency",
            "carrier",
            "e911.carrier-a.net",
            ["PCMU", "PCMA"],
            4,
            0,
            note="E911 trunk; keep enabled",
        ),
        Carrier(
            "e911", "e911.carrier-a.net", ["198.51.100.50"], ["PCMU", "PCMA"], emergency_only=True
        ),
    )
    p.add_trunk(Trunk("intersite", "intersite", "pbx-hq<->pbx-br", ["G722", "PCMU", "PCMA"], 10, 2))

    p.dns_srv = {
        "_sip._udp.carrier-a.net": ("edge1.carrier-a.net", SIP_PORT),
        "_sip._udp.carrier-b.net": ("sbc.carrier-b.net", SIP_PORT),
    }
    p.dns_a = {
        "edge1.carrier-a.net": "198.51.100.10",
        "sbc.carrier-b.net": "203.0.113.20",
        "e911.carrier-a.net": "198.51.100.50",
        "pbx-hq.voice.example.com": "10.10.0.5",
        "pbx-br.voice.example.com": "10.20.0.5",
        "sbc1.voice.example.com": "10.10.0.2",
    }
    for cred in (
        Credential("cred-carrier-a", "carrier-a.net", "acme-voice"),
        Credential("cred-carrier-a-2023", "carrier-a.net", "acme-voice", "rotated"),
        Credential("cred-carrier-b", "carrier-b.net", "acme-intl"),
        Credential("cred-carrier-b-2022", "carrier-b.net", "acme-intl", "rotated"),
    ):
        p.credentials[cred.ref] = cred

    # special-purpose phones, present in every task
    p.add_extension(
        Extension(
            "190", "door-entry", "pbx-hq", "2N IP Verso", "10.10.1.90", ["PCMU"], "+14155550190"
        )
    )
    p.add_extension(
        Extension("299", "paging-zone", "pbx-br", "Algo 8301", "10.20.1.99", ["PCMU", "PCMA"])
    )

    # premium-rate ranges always answer: that is exactly why policy blocks them
    p.pstn_ranges = [
        r"\+1900\d{7}",
        r"\+1976\d{7}",
        r"\+1\d{3}976\d{4}",
        r"\+449\d{9}",
        r"\+611900\d{6}",
        r"\+49900\d{7}",
    ]
    return p
