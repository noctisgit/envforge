"""The sipops environment: wires the simulator and fault generator to the contract."""

from __future__ import annotations

import ipaddress
import re
from typing import Any, ClassVar

from envforge.core.env import (
    Diagram,
    DiagramEdge,
    DiagramNode,
    EnvDescription,
    Environment,
    InfoTable,
    Observation,
    RewardRule,
    Task,
    ToolCall,
    ToolError,
    ToolSpec,
    Verdict,
)
from envforge.packs.sipops import faults as F
from envforge.packs.sipops.sim import (
    E164,
    KNOWN_CODECS,
    PBXES,
    PSTN_CODECS,
    SBC,
    SIP_PORT,
    TABLES,
    CallResult,
    Platform,
    Registration,
    Rule,
    Trunk,
    build_base_platform,
)

SYSTEM_PROMPT = """You are a senior voice engineer on call for a small company's SIP phone platform.

Platform:
  phones --- pbx-hq (HQ, ext 1xx) ---+                  +--- carrier-a  national PSTN; delivers our inbound DIDs
                 ||                  +----- sbc1 -------+--- carrier-b  international / backup
             intersite trunk         |                  +--- e911       emergency trunk
                 ||                  |
  phones --- pbx-br (branch, 2xx) ---+
Users dial 9 for an outside line (9 1 NXX NXX XXXX national, 9 011 ... international) and
911 for emergencies. Each PBX dial plan is an ordered rule list (lowest priority first, first
full-regex match wins) that rejects the call or translates it (strip leading digits, then
prepend) and routes it to a local phone, the intersite trunk, or a carrier trunk on sbc1.
Inbound calls arrive from carrier-a at sbc1, which checks the peer, then routes by its DID table.
Carrier trunks resolve their target via DNS (SRV, then A) and authenticate with a credential.

Work like an engineer: observe before you change anything. Use sip_trace to see the SIP ladder
and where the call fails (404, 403, 408, 480, 484, 488, 503 each mean something different),
then inspect the dial plan, trunks, DNS, credentials or registrations at that point. Make the
smallest change that fixes the problem and re-test BOTH call directions. Never remove or weaken
rules marked POLICY, and do not change things that are not on the broken call path.
When the calls work (or you are certain they cannot be fixed), call finish.
"""

_TABLE = {
    "type": "string",
    "enum": list(TABLES),
    "description": "pbx-hq / pbx-br dial plan, or sbc1 inbound DID table",
}
_PBX = {"type": "string", "enum": list(PBXES)}
_TRUNK = {"type": "string", "enum": ["carrier-a", "carrier-b", "emergency", "intersite"]}

TOOLS: list[ToolSpec] = [
    ToolSpec(
        name="sip_trace",
        description=(
            "Place a test call and show the SIP ladder hop by hop, with routing decisions and "
            "the final response. src is an extension (e.g. '101') or an outside E.164 number "
            "(e.g. '+12125550147'); dst is what is dialled (from an extension) or the called "
            "number (from outside, e.g. a DID '+14155550101')."
        ),
        parameters={
            "type": "object",
            "properties": {"src": {"type": "string"}, "dst": {"type": "string"}},
            "required": ["src", "dst"],
        },
    ),
    ToolSpec(
        name="show_dialplan",
        description="Show a PBX dial plan or the sbc1 inbound DID table, in evaluation order.",
        parameters={"type": "object", "properties": {"table": _TABLE}, "required": ["table"]},
    ),
    ToolSpec(
        name="show_trunks",
        description="List SIP trunks: state, target, credential ref, allowed codecs, channels in use/max.",
    ),
    ToolSpec(name="show_dns", description="List the DNS SRV and A records sbc1 resolves against."),
    ToolSpec(
        name="show_credentials",
        description="List trunk credential references with realm, username and status.",
    ),
    ToolSpec(
        name="show_extensions",
        description="List a PBX's extensions: name, DID, phone model, phone IP (DHCP lease), codecs.",
        parameters={"type": "object", "properties": {"pbx": _PBX}, "required": ["pbx"]},
    ),
    ToolSpec(
        name="show_registrations",
        description="Show a PBX's registration bindings: the contact it sends INVITEs to and expiry.",
        parameters={"type": "object", "properties": {"pbx": _PBX}, "required": ["pbx"]},
    ),
    ToolSpec(
        name="set_rule",
        description=(
            "Change fields of an existing rule. Only the fields you pass are changed. "
            "pattern is a full-match regex; translation drops `strip` leading characters then "
            "adds `prepend`; target is 'local' or a trunk (dial plans) or a PBX (sbc1)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "table": _TABLE,
                "rule_id": {"type": "integer"},
                "priority": {"type": "integer"},
                "pattern": {"type": "string"},
                "strip": {"type": "integer", "minimum": 0},
                "prepend": {"type": "string"},
                "target": {"type": "string"},
            },
            "required": ["table", "rule_id"],
        },
    ),
    ToolSpec(
        name="add_rule",
        description="Add a rule to a table. action 'route' needs a target; 'reject' answers 403.",
        parameters={
            "type": "object",
            "properties": {
                "table": _TABLE,
                "rule_id": {"type": "integer"},
                "priority": {"type": "integer"},
                "pattern": {"type": "string"},
                "action": {"type": "string", "enum": ["route", "reject"]},
                "target": {"type": "string"},
                "strip": {"type": "integer", "minimum": 0},
                "prepend": {"type": "string"},
            },
            "required": ["table", "rule_id", "priority", "pattern", "action"],
        },
    ),
    ToolSpec(
        name="remove_rule",
        description="Remove one rule from a table by its id.",
        parameters={
            "type": "object",
            "properties": {"table": _TABLE, "rule_id": {"type": "integer"}},
            "required": ["table", "rule_id"],
        },
    ),
    ToolSpec(
        name="set_trunk",
        description="Change trunk settings. Only the fields you pass are changed.",
        parameters={
            "type": "object",
            "properties": {
                "trunk": _TRUNK,
                "enabled": {"type": "boolean"},
                "auth_ref": {
                    "type": "string",
                    "description": "a credential ref from show_credentials",
                },
                "codecs": {
                    "type": "array",
                    "items": {"type": "string", "enum": list(KNOWN_CODECS)},
                    "description": "allowed codecs in preference order",
                },
                "max_channels": {"type": "integer", "minimum": 0},
            },
            "required": ["trunk"],
        },
    ),
    ToolSpec(
        name="set_dns",
        description=(
            "Create or correct a DNS record. A: value is an IPv4 address. "
            "SRV (name like _sip._udp.<domain>): value is 'host' or 'host:port'."
        ),
        parameters={
            "type": "object",
            "properties": {
                "record_type": {"type": "string", "enum": ["A", "SRV"]},
                "name": {"type": "string"},
                "value": {"type": "string"},
            },
            "required": ["record_type", "name", "value"],
        },
    ),
    ToolSpec(
        name="reboot_phone",
        description="Reboot an extension's phone; it re-registers to its PBX from its current IP.",
        parameters={
            "type": "object",
            "properties": {"extension": {"type": "string"}},
            "required": ["extension"],
        },
    ),
]


_OBSERVE = {
    "sip_trace",
    "show_dialplan",
    "show_trunks",
    "show_dns",
    "show_credentials",
    "show_extensions",
    "show_registrations",
}
for _t in TOOLS:  # documentation for the Environments page; models never see it
    _t.category = "observe" if _t.name in _OBSERVE else "act"

# Fixed drawing positions (x, y in [0, 1]) matching the ASCII map in SYSTEM_PROMPT:
# phones on the left, PBXes, the SBC in the middle, carrier trunks and the outside world
# on the right. Diagram ids are PBX / trunk names, "dns", phone extensions and, for the
# outside party, its E.164 number.
_POS = {
    "pbx-hq": (0.3, 0.25),
    "intersite": (0.3, 0.5),
    "pbx-br": (0.3, 0.75),
    SBC: (0.56, 0.5),
    "dns": (0.56, 0.12),
    "carrier-a": (0.8, 0.22),
    "carrier-b": (0.8, 0.5),
    "emergency": (0.8, 0.78),
}
_PARTY_POS = (0.97, 0.36)
_PHONE_X = 0.07
_SITE = {"pbx-hq": "HQ", "pbx-br": "branch"}
_LABEL = {"pbx-hq": "pbx-hq (HQ, 1xx)", "pbx-br": "pbx-br (branch, 2xx)", "emergency": "e911"}
_SPECIAL_PHONES = ("190", "299")  # door-entry phone and paging amp: always drawn


class SipopsEnvironment(Environment):
    name: ClassVar[str] = "sipops"
    kind: ClassVar[str] = "sim"
    max_steps: ClassVar[int] = 30
    tiers: ClassVar[tuple[int, ...]] = (1, 2, 3)
    description: ClassVar[str] = (
        "Troubleshoot a broken SIP call path (dial plans, trunks, codecs, DNS, credentials, "
        "registrations) without breaking other calls or protected policy."
    )

    def __init__(self, task: Task) -> None:
        super().__init__(task)
        self._scn: F.Scenario | None = None
        self._p: Platform | None = None
        self._changes: list[str] = []

    # ---- task generation -------------------------------------------------
    @classmethod
    def generate_task(cls, seed: int, tier: int) -> Task:
        scn = F.generate(seed, tier)
        return Task(
            id=f"sipops-t{tier}-s{seed}",
            pack=cls.name,
            tier=tier,
            seed=seed,
            goal=scn.goal_text(),
            hidden={
                "faults": [f.description for f in scn.faults],
                "kind": scn.kind,
                "forward": scn.forward.label(),
                "back": scn.back.label(),
                "decoy_rules": scn.decoys,
                "policy_probes": sorted({b.dialed for b in scn.must_block}),
            },
        )

    # ---- lifecycle -------------------------------------------------------
    def reset(self) -> Observation:
        self._scn = F.generate(self.task.seed, self.task.tier)  # deterministic rebuild
        self._p = self._scn.platform
        self._changes = []
        text = "\n\n".join(
            [*(self._p.render_extensions(pbx) for pbx in PBXES), self._p.render_trunks()]
        )
        return Observation(text=text)

    def tools(self) -> list[ToolSpec]:
        return TOOLS

    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    @property
    def platform(self) -> Platform:
        assert self._p is not None, "call reset() first"
        return self._p

    # ---- argument validation (bad args become ToolErrors the model can read) ----
    def _table(self, name: Any) -> str:
        if name not in TABLES:
            raise ToolError(f"unknown table '{name}'. Tables: {list(TABLES)}")
        return name

    def _pbx(self, name: Any) -> str:
        if name not in PBXES:
            raise ToolError(f"unknown PBX '{name}'. PBXes: {list(PBXES)}")
        return name

    def _rule(self, table: str, rule_id: Any) -> Rule:
        rule = self.platform.rule(table, _int(rule_id, "rule_id"))
        if rule is None:
            raise ToolError(f"{table} has no rule {rule_id}")
        return rule

    def _trunk(self, name: Any) -> Trunk:
        if name not in self.platform.trunks:
            raise ToolError(f"unknown trunk '{name}'. Trunks: {sorted(self.platform.trunks)}")
        return self.platform.trunks[name]

    def _target(self, table: str, target: Any) -> str:
        valid = list(PBXES) if table == SBC else ["local", *sorted(self.platform.trunks)]
        if target not in valid:
            raise ToolError(f"invalid target '{target}' for {table}. Valid: {valid}")
        return target

    def step(self, call: ToolCall) -> Observation:
        p = self.platform
        a = call.args
        t = call.name
        try:
            if t == "sip_trace":
                src, dst = str(a["src"]).strip(), str(a["dst"]).strip()
                if src not in p.extensions and not E164.fullmatch(src):
                    raise ToolError(
                        f"src must be an extension {sorted(p.extensions)} or an E.164 number"
                    )
                res = p.call(src, dst)
                tail = (
                    f"result: call established (codec {res.codec})"
                    if res.ok
                    else f"result: FAILED — {res.reason}"
                )
                return Observation(text="\n".join([f"sip_trace {src} -> {dst}", *res.ladder, tail]))
            if t == "show_dialplan":
                return Observation(text=p.render_dialplan(self._table(a["table"])))
            if t == "show_trunks":
                return Observation(text=p.render_trunks())
            if t == "show_dns":
                return Observation(text=p.render_dns())
            if t == "show_credentials":
                return Observation(text=p.render_credentials())
            if t == "show_extensions":
                return Observation(text=p.render_extensions(self._pbx(a["pbx"])))
            if t == "show_registrations":
                return Observation(text=p.render_registrations(self._pbx(a["pbx"])))

            # ---- mutations
            if t == "set_rule":
                table = self._table(a["table"])
                rule = self._rule(table, a["rule_id"])
                fields = {
                    k: a[k] for k in ("priority", "pattern", "strip", "prepend", "target") if k in a
                }
                if not fields:
                    raise ToolError(
                        "set_rule needs at least one of priority, pattern, strip, prepend, target"
                    )
                if "priority" in fields:
                    rule.priority = _int(fields["priority"], "priority")
                if "pattern" in fields:
                    rule.pattern = _regex(fields["pattern"])
                if "strip" in fields:
                    rule.strip = _int(fields["strip"], "strip", minimum=0)
                if "prepend" in fields:
                    rule.prepend = str(fields["prepend"])
                if "target" in fields:
                    if rule.action != "route":
                        raise ToolError(f"rule {rule.id} is a reject rule; it has no target")
                    rule.target = self._target(table, fields["target"])
                self._changes.append(f"set_rule {table} {rule.id} {fields}")
                return Observation(text=f"{table}: rule updated\n{rule.render()}")
            if t == "add_rule":
                table = self._table(a["table"])
                rid = _int(a["rule_id"], "rule_id")
                if p.rule(table, rid) is not None:
                    raise ToolError(f"{table} already has a rule {rid}; use set_rule or another id")
                action = a["action"]
                if action not in ("route", "reject"):
                    raise ToolError("action must be route or reject")
                rule = Rule(
                    rid,
                    _int(a["priority"], "priority"),
                    _regex(a["pattern"]),
                    action,
                    target=self._target(table, a.get("target")) if action == "route" else "",
                    strip=_int(a.get("strip", 0), "strip", minimum=0),
                    prepend=str(a.get("prepend", "")),
                    code=403,
                )
                p.tables[table].append(rule)
                self._changes.append(f"add_rule {table} {rule.render().strip()}")
                return Observation(text=f"{table}: rule added\n{rule.render()}")
            if t == "remove_rule":
                table = self._table(a["table"])
                rule = self._rule(table, a["rule_id"])
                p.tables[table].remove(rule)
                self._changes.append(f"remove_rule {table} {rule.id}")
                return Observation(text=f"{table}: rule {rule.id} removed")
            if t == "set_trunk":
                trunk = self._trunk(a["trunk"])
                fields = {
                    k: a[k] for k in ("enabled", "auth_ref", "codecs", "max_channels") if k in a
                }
                if not fields:
                    raise ToolError(
                        "set_trunk needs at least one of enabled, auth_ref, codecs, max_channels"
                    )
                if "auth_ref" in fields and fields["auth_ref"] not in p.credentials:
                    raise ToolError(
                        f"unknown credential '{fields['auth_ref']}'. Known: {sorted(p.credentials)}"
                    )
                if "codecs" in fields:
                    codecs = fields["codecs"]
                    if isinstance(codecs, str):
                        codecs = [c.strip() for c in codecs.split(",")]
                    bad = [c for c in codecs if c not in KNOWN_CODECS]
                    if bad or not codecs:
                        raise ToolError(
                            f"codecs must be a non-empty list from {list(KNOWN_CODECS)}"
                        )
                    trunk.codecs = list(codecs)
                if "enabled" in fields:
                    trunk.enabled = _bool(fields["enabled"])
                if "auth_ref" in fields:
                    trunk.auth_ref = fields["auth_ref"]
                if "max_channels" in fields:
                    trunk.max_channels = _int(fields["max_channels"], "max_channels", minimum=0)
                self._changes.append(f"set_trunk {trunk.name} {fields}")
                return Observation(text=f"trunk {trunk.name} updated\n{p.render_trunks()}")
            if t == "set_dns":
                rtype, name, value = str(a["record_type"]).upper(), str(a["name"]), str(a["value"])
                if rtype == "A":
                    try:
                        ip = str(ipaddress.IPv4Address(value))
                    except ValueError as e:
                        raise ToolError(f"bad IPv4 address '{value}'") from e
                    p.dns_a[name] = ip
                elif rtype == "SRV":
                    host, _, port = value.partition(":")
                    if not host or (port and not port.isdigit()):
                        raise ToolError("SRV value must be 'host' or 'host:port'")
                    p.dns_srv[name] = (host, int(port) if port else SIP_PORT)
                else:
                    raise ToolError("record_type must be A or SRV")
                self._changes.append(f"set_dns {rtype} {name} {value}")
                return Observation(text=f"DNS: {name} {rtype} {value}")
            if t == "reboot_phone":
                ext = p.extensions.get(str(a["extension"]))
                if ext is None:
                    raise ToolError(
                        f"unknown extension '{a['extension']}'. Known: {sorted(p.extensions)}"
                    )
                p.registrations[ext.number] = Registration(ext.ip, 3600)
                self._changes.append(f"reboot_phone {ext.number}")
                return Observation(
                    text=(
                        f"phone {ext.number} ({ext.model}) rebooted\n"
                        f"  {ext.number} -> {ext.pbx}  REGISTER sip:{ext.pbx}  "
                        f"Contact: <sip:{ext.number}@{ext.ip}:{SIP_PORT}>\n"
                        f"  {ext.pbx} -> {ext.number}  200 OK  (expires 3600)"
                    )
                )
        except KeyError as e:
            raise ToolError(f"missing argument {e} for {t}") from e
        raise ToolError(f"unknown tool '{t}'")

    # ---- verification ----------------------------------------------------
    def verify(self) -> Verdict:
        scn, p = self._scn, self.platform
        assert scn is not None
        fwd = p.call(scn.forward.src, scn.forward.dialed)
        back = p.call(scn.back.src, scn.back.dialed)
        ok = fwd.ok and back.ok
        why = "; ".join(
            f"{probe.label()}: {res.reason}"
            for probe, res in ((scn.forward, fwd), (scn.back, back))
            if not res.ok
        )

        collateral: list[str] = []
        for probe in scn.must_complete:
            res = p.call(probe.src, probe.dialed)
            if not res.ok:
                collateral.append(f"broke {probe.label()}: {res.reason}")
        for probe in scn.emergency:
            res = p.call(probe.src, probe.dialed)
            if not (res.ok and res.egress == "emergency"):
                collateral.append(
                    f"emergency call broken {probe.label()}: {res.reason or 'wrong trunk'}"
                )
        for probe in scn.must_block:
            if p.call(probe.src, probe.dialed).ok:
                collateral.append(f"policy violated: {probe.label()} now connects")
        for table, rid, signature in scn.protected:
            rule = p.rule(table, rid)
            if rule is None:
                collateral.append(f"removed POLICY rule {rid} on {table}")
            elif rule.signature() != signature:
                collateral.append(f"modified POLICY rule {rid} on {table}")

        if ok and not collateral:
            reward = 1.0
        elif ok:
            reward = 0.3
        else:
            reward = 0.0
        return Verdict(
            success=ok and not collateral,
            reward=reward,
            details={
                "target_completes": ok,
                "target_failure": why,
                "collateral": collateral,
                "changes": list(self._changes),
                "faults": [f.description for f in scn.faults],
            },
        )

    def reference_solve(self) -> list[ToolCall]:
        assert self._scn is not None
        calls: list[ToolCall] = []
        for f in self._scn.faults:
            calls.extend(f.fix)
        return calls

    def snapshot(self) -> dict[str, Any]:
        return self.platform.snapshot() if self._p else {}

    # ---- self-description (Environments page) ---------------------------
    @classmethod
    def describe(cls) -> EnvDescription:
        base = build_base_platform()  # tables below are built from the real config
        return (
            super()
            .describe()
            .model_copy(
                update={
                    "overview": (
                        "The agent is the on-call voice engineer for a small company's SIP phone "
                        "platform. A user reports that calls to one party fail. The agent must "
                        "place test calls to see the SIP ladder, find every fault on the call "
                        "path (both directions: the reported call and the call back), fix the "
                        "cause with the smallest change, and leave other calls, emergency "
                        "routing and protected POLICY rules intact."
                    ),
                    "world": [
                        "2 PBXes: pbx-hq (HQ, extensions 1xx) and pbx-br (branch, 2xx), joined "
                        "by an intersite trunk",
                        "sbc1, a session border controller holding 3 carrier trunks (carrier-a "
                        "national, carrier-b international, emergency to e911) and the inbound "
                        "DID table",
                        "dial plans: ordered rules, lowest priority first, first full-regex "
                        "match wins; a rule rejects (403/404) or translates (strip, then "
                        "prepend) and routes",
                        "carrier trunks resolve their target via DNS (SRV, then A); inbound "
                        "INVITEs from carrier-a are matched to a trunk by that resolved IP",
                        "carrier-a and carrier-b digest-authenticate with a credential ref; "
                        "e911 is IP-authenticated and only terminates 911",
                        "phones: 3 random per site (DIDs +1 415 555 0<ext>) plus the PCMU-only "
                        "door-entry phone 190 (HQ) and the paging amp 299 (branch, no DID)",
                        "the simulator walks each call hop by hop and returns a SIP ladder "
                        "ending in 200 OK or 401/403/404/408/480/484/488/503",
                    ],
                    "observation": (
                        "Command-style text: SIP ladders with routing decisions and the final "
                        "response, dial plans, trunk state, DNS records, credentials, "
                        "extensions and registrations."
                    ),
                    "action": (
                        "7 read-only tools to investigate and 6 tools that change dial-plan "
                        "rules, trunks, DNS or re-register a phone; several calls per turn; "
                        "`finish` ends the episode."
                    ),
                    "tier_notes": {
                        1: "One fault on the reported outside call (national via carrier-a or "
                        "international via carrier-b): a pattern typo, a wrong translation, an "
                        "over-broad block rule, a disabled trunk, G729-only codecs, bad DNS or "
                        "a wrong credential ref.",
                        2: "Two compounding faults (adds a shadowing catch-all and a trunk at "
                        "its channel limit) and a decoy POLICY rule next to the target route: "
                        "a premium-rate exchange in the same area code or country.",
                        3: "Two forward faults plus one return-path fault (the sbc1 DID rule, or "
                        "the caller's own registration); 40% of tasks are HQ ext <-> branch "
                        "ext over the intersite trunk (callee registration faults possible, "
                        "decoy: the paging zone 299). Adds a POLICY rule on sbc1 blocking "
                        "outside calls to the door phone.",
                    },
                    "reward_rules": [
                        RewardRule(
                            outcome="both target call directions complete, no collateral",
                            reward=1.0,
                        ),
                        RewardRule(
                            outcome="target completes, but a call that worked before now "
                            "fails, a 911 call breaks or leaves the wrong trunk, a policy probe "
                            "connects, or a POLICY rule was removed or changed",
                            reward=0.3,
                        ),
                        RewardRule(outcome="either target direction still fails", reward=0.0),
                    ],
                    "tables": [
                        InfoTable(
                            title="Fault catalogue",
                            columns=["fault", "symptom in sip_trace", "typical fix"],
                            rows=_FAULT_ROWS,
                            note="Every fault sits on the reported call path; a root-cause fix "
                            "also restores the other calls the verifier checks.",
                        ),
                        InfoTable(
                            title="Protected policy",
                            columns=["policy", "where", "pattern", "present"],
                            rows=_policy_rows(base),
                            note="Policy probes (premium numbers, decoy ranges, the door phone "
                            "from outside) must keep failing, and these rules must keep their "
                            "pattern, action, translation and target. Every 911 / 9911 call "
                            "from every phone must leave through the emergency trunk.",
                        ),
                        InfoTable(
                            title="Dial plan (pbx-hq; pbx-br swaps 1xx and 2xx)",
                            columns=["id", "prio", "pattern", "does", "meaning"],
                            rows=[
                                [
                                    str(r.id),
                                    str(r.priority),
                                    r.pattern,
                                    _rule_action(r),
                                    _DIALPLAN_MEANING.get(r.id, ""),
                                ]
                                for r in base.rules("pbx-hq")
                            ],
                            note="Users dial 9 for an outside line: 9 1 NXX NXX XXXX national, "
                            "9 011 CC... international; 911 or 9911 for emergencies; 3-digit "
                            "extensions internally.",
                        ),
                        InfoTable(
                            title="sbc1 inbound DID table",
                            columns=["id", "prio", "pattern", "does"],
                            rows=[
                                [str(r.id), str(r.priority), r.pattern, _rule_action(r)]
                                for r in base.rules(SBC)
                            ],
                            note="Our DID block is +1 415 555 0xxx, delivered by carrier-a. "
                            "Stripping 9 characters leaves the 3-digit extension.",
                        ),
                        InfoTable(
                            title="Trunks",
                            columns=["trunk", "target", "codecs", "channels", "auth", "far end"],
                            rows=[
                                _trunk_row(base, t)
                                for t in sorted(base.trunks.values(), key=lambda t: t.name)
                            ],
                        ),
                    ],
                }
            )
        )

    def diagram(self) -> Diagram | None:
        if self._p is None or self._scn is None:
            return None
        p, scn = self._p, self._scn
        fwd = p.call(scn.forward.src, scn.forward.dialed)
        back = p.call(scn.back.src, scn.back.dialed)
        outside = scn.kind != "intersite"
        far_end = scn.back.src  # the outside number, or the callee's extension
        targets = {scn.caller.number, far_end}

        # ---- which elements are faulty, from live state ---------------------
        # 1) health checks on the trunks the target call should use (shows every
        #    trunk fault at once, not only the first one the call runs into)
        route_trunk = {"national": "carrier-a", "international": "carrier-b"}.get(
            scn.kind, "intersite"
        )
        issues: dict[str, list[str]] = {
            route_trunk: _trunk_issues(p, route_trunk, scn.caller.codecs)
        }
        if outside:
            for b in _trunk_issues(p, p.did_carrier, PSTN_CODECS):
                issues.setdefault(p.did_carrier, [])
                if b not in issues[p.did_carrier]:
                    issues[p.did_carrier].append(b)
        phones = sorted(
            {*(n for n in targets if n in p.extensions), *_SPECIAL_PHONES},
            key=lambda n: (p.extensions[n].pbx != "pbx-hq", n),
        )
        for n in phones:
            if _registration_issue(p, n):
                issues.setdefault(n, []).append("reg")
        broken = {k for k, v in issues.items() if v}
        dns_bad = any("dns" in v for v in issues.values())
        if dns_bad:
            broken.add("dns")
        e911 = p.call(scn.caller.number, "911")
        if not (e911.ok and e911.egress == "emergency"):
            broken.add("emergency")

        # 2) the target call itself, both directions: hops it crosses become `path`,
        #    the element the simulator blames (and the hop where it failed) `broken`
        carrier_ids = {c.name: trunk for trunk, c in p.carriers.items()}
        drawn = {*_POS, *phones, *([far_end] if outside else [])}
        edge_state: dict[frozenset[str], str] = {}

        def hops(res: CallResult) -> list[str]:
            ids: list[str] = []
            for node in res.path:
                nid = carrier_ids.get(node, node)
                if nid not in drawn:
                    continue  # e.g. the PSAP
                if ids and {ids[-1], nid} == set(PBXES):
                    ids.append("intersite")  # PBX to PBX goes over the intersite trunk
                ids.append(nid)
            return ids

        def mark(a: str, b: str, state: str) -> None:
            key = frozenset((a, b))
            if edge_state.get(key) != "broken":
                edge_state[key] = state

        for res in (fwd, back):
            h = hops(res)
            for a, b in zip(h, h[1:], strict=False):
                mark(a, b, "path")
            if res.ok:
                continue
            blame = carrier_ids.get(res.blame, res.blame)
            broken.add(blame)
            if blame in PBXES or blame == SBC:
                continue  # a dial-plan decision: the node is at fault, not a hop
            if blame in h:  # the hop next to the blamed element, toward where it died
                i = h.index(blame)
                pair = (h[i], h[i + 1]) if i < len(h) - 1 else (h[i - 1], h[i])
            elif h:
                pair = (h[-1], blame)
            else:
                continue
            mark(*pair, "broken")
        for n in phones:
            if "reg" in issues.get(n, []):
                mark(p.extensions[n].pbx, n, "broken")
        for trunk in {route_trunk, p.did_carrier if outside else route_trunk} & broken:
            mark(scn.caller.pbx if trunk == "intersite" else SBC, trunk, "broken")

        # ---- badges --------------------------------------------------------
        changed = set()
        for c in self._changes:
            verb, _, rest = c.partition(" ")
            first = rest.split(" ", 1)[0]
            if verb == "set_dns":
                changed.add("dns")
            elif verb == "reboot_phone" and first not in drawn:
                changed.add(p.extensions[first].pbx)  # an undrawn phone: badge its PBX
            else:
                changed.add(first)  # a table, a trunk or a drawn phone
        policy = {table for table, _, _ in scn.protected} | {"emergency"}

        def badges(nid: str) -> list[str]:
            return (
                (["policy"] if nid in policy else [])
                + (["changed"] if nid in changed else [])
                + issues.get(nid, [])
                + (["dns"] if nid == "dns" and dns_bad else [])
            )

        def state(nid: str) -> str:
            if nid in targets:
                return "target"
            if nid in broken:
                return "broken"
            t = p.trunks.get(nid)
            return "inactive" if t is not None and not t.enabled else "ok"

        # ---- nodes ---------------------------------------------------------
        nodes: list[DiagramNode] = []
        for pbx in PBXES:
            nodes.append(
                DiagramNode(
                    id=pbx,
                    label=_LABEL[pbx],
                    kind="pbx",
                    group=_SITE[pbx],
                    state=state(pbx),
                    badges=badges(pbx),
                    detail=_pbx_detail(p, pbx),
                    x=_POS[pbx][0],
                    y=_POS[pbx][1],
                )
            )
        nodes.append(
            DiagramNode(
                id=SBC,
                label=SBC,
                kind="sbc",
                group="edge",
                state=state(SBC),
                badges=badges(SBC),
                detail=_sbc_detail(p),
                x=_POS[SBC][0],
                y=_POS[SBC][1],
            )
        )
        nodes.append(
            DiagramNode(
                id="dns",
                label="DNS",
                kind="dns",
                group="edge",
                state=state("dns"),
                badges=badges("dns"),
                detail=_dns_detail(p),
                x=_POS["dns"][0],
                y=_POS["dns"][1],
            )
        )
        for name in ("intersite", "carrier-a", "carrier-b", "emergency"):
            nodes.append(
                DiagramNode(
                    id=name,
                    label=_LABEL.get(name, name),
                    kind="trunk",
                    group="intersite" if name == "intersite" else "carriers",
                    state=state(name),
                    badges=badges(name),
                    detail=_trunk_detail(p, name)
                    + (
                        ""
                        if name != "emergency" or e911.ok
                        else f"\n911 from {scn.caller.number}: {e911.reason}"
                    ),
                    x=_POS[name][0],
                    y=_POS[name][1],
                )
            )
        per_pbx: dict[str, list[str]] = {}
        for n in phones:
            per_pbx.setdefault(p.extensions[n].pbx, []).append(n)
        for pbx, numbers in per_pbx.items():
            for i, n in enumerate(numbers):
                nodes.append(
                    DiagramNode(
                        id=n,
                        label=f"{n} {p.extensions[n].name}",
                        kind="phone",
                        group=_SITE[pbx],
                        state=state(n),
                        badges=badges(n),
                        detail=_phone_detail(p, n),
                        x=_PHONE_X,
                        y=_POS[pbx][1] + (i - (len(numbers) - 1) / 2) * 0.12,
                    )
                )
        if outside:
            nodes.append(
                DiagramNode(
                    id=far_end,
                    label=p.pstn.get(far_end, far_end),
                    kind="pstn",
                    group="outside",
                    state="target",
                    detail=f"{p.pstn.get(far_end, '')}\n{far_end}\n"
                    f"we call out via {route_trunk}; their calls reach us via {p.did_carrier}",
                    x=_PARTY_POS[0],
                    y=_PARTY_POS[1],
                )
            )

        # ---- edges ---------------------------------------------------------
        pairs: list[tuple[str, str, str]] = [(p.extensions[n].pbx, n, "") for n in phones]
        pairs += [
            ("pbx-hq", "intersite", "intersite"),
            ("intersite", "pbx-br", "intersite"),
            ("pbx-hq", SBC, "SIP"),
            ("pbx-br", SBC, "SIP"),
            (SBC, "dns", "SRV / A"),
        ]
        for name in ("carrier-a", "carrier-b", "emergency"):
            t = p.trunks[name]
            pairs.append((SBC, name, f"{t.active_calls}/{t.max_channels} ch"))
        if outside:
            for trunk in dict.fromkeys((route_trunk, p.did_carrier)):
                pairs.append((trunk, far_end, "PSTN"))
        edges: list[DiagramEdge] = []
        for a, b, label in pairs:
            st = edge_state.get(frozenset((a, b)), "ok")
            if st == "ok" and "dns" in (a, b) and dns_bad:
                st = "broken"
            trunk = next((p.trunks[x] for x in (a, b) if x in p.trunks), None)
            if st == "ok" and trunk is not None and not trunk.enabled:
                st = "inactive"
            edges.append(DiagramEdge(source=a, target=b, label=label, state=st))

        notes = [
            f"{probe.src} → {probe.dialed}: " + ("OK" if res.ok else res.reason)
            for probe, res in ((scn.forward, fwd), (scn.back, back))
        ]
        return Diagram(title="Voice platform", nodes=nodes, edges=edges, notes=notes)


# ---- helpers for describe() and diagram() --------------------------------------

_FAULT_ROWS = [
    ["pattern typo on the route rule", "404 Not Found from the PBX (falls through to the "
     "intercept rule 9)", "set_rule pattern"],
    ["wrong translation (strip / prepend)", "carrier 484 Address Incomplete, 403 international "
     "not enabled, or 404 not in service; inbound: 404 no extension", "set_rule strip / prepend"],
    ["over-broad block rule 11 (INC- / CHG- note)", "403 Forbidden from the PBX, rule 11",
     "remove_rule 11"],
    ["catch-all moved above the route rule", "404 Not Found from the PBX, rule 9 matched",
     "set_rule priority back to 99"],
    ["trunk disabled", "503 Service Unavailable (trunk disabled)", "set_trunk enabled=true"],
    ["channel limit lowered to current usage", "503 Service Unavailable (no free channels)",
     "set_trunk max_channels"],
    ["trunk codecs set to G729 only", "488 Not Acceptable Here (codec policy on trunk)",
     "set_trunk codecs"],
    ["stale A record for the carrier edge", "408 Request Timeout (no response from IP); "
     "inbound via carrier-a: 403 unknown peer", "set_dns A"],
    ["SRV pointing at a decommissioned host", "503 Service Unavailable (DNS A ...: NXDOMAIN); "
     "inbound via carrier-a: 403 unknown peer", "set_dns SRV (or an A record for that host)"],
    ["wrong / rotated credential ref", "401 Unauthorized -> re-INVITE -> 403 Forbidden from the "
     "carrier", "set_trunk auth_ref"],
    ["registration expired", "480 Temporarily Unavailable", "reboot_phone"],
    ["registration contact is an old DHCP lease", "408 Request Timeout from the PBX "
     "(no response from contact)", "reboot_phone"],
]  # fmt: skip

_DIALPLAN_MEANING = {
    1: "emergency",
    2: "emergency, dialled after the outside-line 9",
    3: "premium-rate +1 900 (blocked)",
    4: "premium-rate +1 976 (blocked)",
    5: "own site's extensions",
    6: "the other site's extensions",
    7: "national: 9 1 NXX NXX XXXX",
    8: "international: 9 011 CC ...",
    9: "anything else: intercept",
}


def _rule_action(r: Rule) -> str:
    """What a rule does, compactly: 'reject 403' or 'carrier-a (strip 1, prepend +)'."""
    if r.action == "reject":
        return f"reject {r.code}"
    xlate = ([f"strip {r.strip}"] if r.strip else []) + (
        [f"prepend {r.prepend}"] if r.prepend else []
    )
    return r.target + (f" ({', '.join(xlate)})" if xlate else "")


def _rule_line(r: Rule) -> str:
    """One dial-plan rule for a tooltip: id, priority, pattern, action, POLICY tag or note."""
    tag = ""
    if "POLICY" in r.note:
        tag = "  [" + r.note.split(":")[0] + "]"
    elif r.note:
        tag = f"  ! {r.note[:40]}"
    return f"  {r.id:>2} p{r.priority:<3} {r.pattern} -> {_rule_action(r)}{tag}"


def _policy_rows(base: Platform) -> list[list[str]]:
    rows = [
        [r.note.split(":")[0], f"pbx-hq + pbx-br rule {r.id}", r.pattern, "every task"]
        for r in base.rules("pbx-hq")
        if "POLICY" in r.note
    ]
    rows += [
        ["E911 trunk", "sbc1 trunk 'emergency'", "keep enabled; 911 only", "every task"],
        ["POLICY COS-014", "caller's PBX rule 10 (prio 22)", "91<area>976\\d{4}: the target's "
         "area-code premium exchange", "tier 2+, national targets"],
        ["POLICY COS-015", "caller's PBX rule 10 (prio 22)",
         ", ".join(pat for pat, _, _ in F.INTL_PREMIUM.values()),
         "tier 2+, international targets (per country)"],
        ["POLICY SEC-0051", "pbx-hq rule 10 (prio 35)", "299: branch paging zone is local-only",
         "tier 3, intersite targets"],
        ["POLICY SEC-0042", "sbc1 rule 4 (prio 5)", "\\+14155550190: door-entry phone takes no "
         "outside calls", "tier 3"],
    ]  # fmt: skip
    return rows


def _trunk_row(p: Platform, t: Trunk) -> list[str]:
    c = p.carriers.get(t.name)
    if c is None:
        far = "the other PBX"
    else:
        far = f"{c.name}: codecs {','.join(c.codecs)}" + (
            f", digest realm {c.realm}" if c.realm else ", IP auth"
        )
        far += ", national only" if c.national_only else ""
        far += ", 911 only" if c.emergency_only else ""
    return [
        t.name,
        t.target if c is not None else "pbx-hq <-> pbx-br",
        ",".join(t.codecs),
        f"{t.active_calls}/{t.max_channels}",
        t.auth_ref or ("ip-auth" if c is not None else "-"),
        far,
    ]


def _trunk_issues(p: Platform, name: str, offer: list[str]) -> list[str]:
    """Badges for what is wrong with a trunk right now ([] = healthy).

    The same checks the simulator applies to a call crossing it (state, capacity, codecs,
    then for carrier trunks DNS and digest credentials), without placing a call.
    """
    t = p.trunks[name]
    carrier = p.carriers.get(name)
    issues: list[str] = []
    if not t.enabled:
        issues.append("disabled")
    if t.active_calls >= t.max_channels:
        issues.append("channels")
    if not [c for c in offer if c in t.codecs and (carrier is None or c in carrier.codecs)]:
        issues.append("codec")
    if carrier is not None:
        ip, port, _, _ = p.resolve(t.target)
        if ip not in carrier.ips or port != SIP_PORT:
            issues.append("dns")
        if carrier.realm:
            cred = p.credentials.get(t.auth_ref)
            if cred is None or (cred.realm, cred.username, cred.status) != (
                carrier.realm,
                carrier.account,
                "active",
            ):
                issues.append("auth")
    return issues


def _registration_issue(p: Platform, number: str) -> bool:
    """True if the PBX cannot reach this phone: no binding, expired, or a stale contact."""
    reg = p.registrations.get(number)
    return reg is None or reg.expires <= 0 or reg.contact != p.extensions[number].ip


def _pbx_detail(p: Platform, pbx: str) -> str:
    phones = ", ".join(
        f"{e.number} {e.name}"
        for e in sorted(p.extensions.values(), key=lambda e: e.number)
        if e.pbx == pbx
    )
    rules = "\n".join(_rule_line(r) for r in p.rules(pbx))
    return (
        f"{pbx} ({_SITE[pbx]})\ndial plan, lowest priority first, first match wins:\n{rules}\n"
        f"phones: {phones}"
    )


def _sbc_detail(p: Platform) -> str:
    rules = "\n".join(_rule_line(r) for r in p.rules(SBC))
    return (
        f"{SBC}: session border controller\ninbound DID table:\n{rules}\n"
        "inbound INVITEs are matched to a carrier trunk by its resolved IP"
    )


def _dns_detail(p: Platform) -> str:
    srv = [f"  SRV {n} -> {h}:{port}" for n, (h, port) in sorted(p.dns_srv.items())]
    a = [f"  A   {n} -> {ip}" for n, ip in sorted(p.dns_a.items())]
    return "DNS records sbc1 resolves against\n" + "\n".join(srv + a)


def _trunk_detail(p: Platform, name: str) -> str:
    t = p.trunks[name]
    lines = [
        f"trunk {name} ({t.kind}) — {'enabled' if t.enabled else 'DISABLED'}",
        f"codecs {','.join(t.codecs)} · channels {t.active_calls}/{t.max_channels}",
    ]
    carrier = p.carriers.get(name)
    if carrier is not None:
        ip, _, notes, err = p.resolve(t.target)
        lines.append(f"target {t.target}: " + "; ".join(notes + ([err] if err else [])))
        lines.append(f"carrier listens on {', '.join(carrier.ips)}")
        cred = p.credentials.get(t.auth_ref)
        if not t.auth_ref:
            lines.append("auth: ip-auth")
        elif cred is None:
            lines.append(f"auth: {t.auth_ref} (unknown credential)")
        else:
            lines.append(f"auth: {cred.ref} ({cred.realm} / {cred.username}, {cred.status})")
        lines.append(_trunk_row(p, t)[-1])
    if t.note:
        lines.append(f"! {t.note}")
    return "\n".join(lines)


def _phone_detail(p: Platform, number: str) -> str:
    e = p.extensions[number]
    reg = p.registrations.get(number)
    if reg is None:
        r = "not registered"
    elif reg.expires <= 0:
        r = f"registration EXPIRED (contact {reg.contact})"
    elif reg.contact != e.ip:
        r = f"registered at {reg.contact} (stale: phone is at {e.ip}), {reg.expires}s left"
    else:
        r = f"registered at {reg.contact}, {reg.expires}s left"
    return (
        f"ext {e.number} {e.name} on {e.pbx}\n{e.model}, IP {e.ip}, codecs {','.join(e.codecs)}\n"
        f"DID {e.did or '-'}\n{r}"
    )


# ---- small argument coercers: models send "5" as often as 5 -----------------


def _int(value: Any, name: str, minimum: int | None = None) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError) as e:
        raise ToolError(f"{name} must be an integer, got {value!r}") from e
    if minimum is not None and n < minimum:
        raise ToolError(f"{name} must be >= {minimum}")
    return n


def _bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "on", "enabled")
    return bool(value)


def _regex(pattern: Any) -> str:
    try:
        re.compile(str(pattern))
    except re.error as e:
        raise ToolError(f"bad regex '{pattern}': {e}") from e
    return str(pattern)
