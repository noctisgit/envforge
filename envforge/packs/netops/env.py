"""The netops environment: wires the simulator and fault generator to the contract."""

from __future__ import annotations

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
from envforge.packs.netops import faults as F
from envforge.packs.netops.sim import LINKS, SITES, Network, net

SYSTEM_PROMPT = """You are a senior network engineer on call for a small enterprise WAN.

Topology: three sites, each behind one router, plus a core router.
  siteA (R1, 10.1.0.0/24) --- core (R2) --- siteB (R3, 10.3.0.0/24)
                                 |               |
                                 +--- dc (R4, 10.4.0.0/24) ---+
Routing is static. ACLs are evaluated top-down, first match wins, implicit permit.
Hostnames resolve through one DNS table.

Work like an engineer: observe before you change anything. Use ping/traceroute to find
where traffic stops, then inspect routes, links, ACLs or DNS at that point. Make the
smallest change that fixes the problem. Re-test in BOTH directions. Never remove ACL rules
marked as security policy, and do not change things that are not on the broken path.
When connectivity is restored (or you are certain it cannot be), call finish.
"""

_ROUTER = {"type": "string", "enum": ["R1", "R2", "R3", "R4"]}

TOOLS: list[ToolSpec] = [
    ToolSpec(
        name="ping",
        description="Test one-way reachability from a host to a host name or IP. Reports where and why it fails.",
        parameters={
            "type": "object",
            "properties": {"src": {"type": "string"}, "dst": {"type": "string"}},
            "required": ["src", "dst"],
        },
    ),
    ToolSpec(
        name="traceroute",
        description="Show the hop-by-hop path from a host toward a destination, and where it stops.",
        parameters={
            "type": "object",
            "properties": {"src": {"type": "string"}, "dst": {"type": "string"}},
            "required": ["src", "dst"],
        },
    ),
    ToolSpec(
        name="dig",
        description="Resolve a hostname in the DNS table.",
        parameters={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    ),
    ToolSpec(
        name="show_hosts", description="List all hosts, their IPs and which router they sit behind."
    ),
    ToolSpec(name="show_links", description="List router-to-router links and whether each is up."),
    ToolSpec(
        name="show_routes",
        description="Show a router's routing table (connected subnets and static routes).",
        parameters={"type": "object", "properties": {"router": _ROUTER}, "required": ["router"]},
    ),
    ToolSpec(
        name="show_acl",
        description="Show a router's ACL rules in evaluation order.",
        parameters={"type": "object", "properties": {"router": _ROUTER}, "required": ["router"]},
    ),
    ToolSpec(
        name="set_route",
        description="Add or replace a static route on a router: prefix (CIDR) via next-hop router name.",
        parameters={
            "type": "object",
            "properties": {
                "router": _ROUTER,
                "prefix": {"type": "string", "description": "e.g. 10.4.0.0/24"},
                "next_hop": _ROUTER,
            },
            "required": ["router", "prefix", "next_hop"],
        },
    ),
    ToolSpec(
        name="delete_route",
        description="Delete a static route (by prefix) from a router.",
        parameters={
            "type": "object",
            "properties": {"router": _ROUTER, "prefix": {"type": "string"}},
            "required": ["router", "prefix"],
        },
    ),
    ToolSpec(
        name="remove_acl_rule",
        description="Remove one ACL rule from a router by its rule id.",
        parameters={
            "type": "object",
            "properties": {"router": _ROUTER, "rule_id": {"type": "integer"}},
            "required": ["router", "rule_id"],
        },
    ),
    ToolSpec(
        name="add_acl_rule",
        description="Insert an ACL rule at the top of a router's ACL. src/dst are CIDR or 'any'.",
        parameters={
            "type": "object",
            "properties": {
                "router": _ROUTER,
                "rule_id": {"type": "integer"},
                "action": {"type": "string", "enum": ["permit", "deny"]},
                "src": {"type": "string"},
                "dst": {"type": "string"},
            },
            "required": ["router", "rule_id", "action", "src", "dst"],
        },
    ),
    ToolSpec(
        name="set_link",
        description="Set a router-to-router link up or down.",
        parameters={
            "type": "object",
            "properties": {
                "a": _ROUTER,
                "b": _ROUTER,
                "state": {"type": "string", "enum": ["up", "down"]},
            },
            "required": ["a", "b", "state"],
        },
    ),
    ToolSpec(
        name="set_dns",
        description="Set or correct a DNS A record: hostname -> IP.",
        parameters={
            "type": "object",
            "properties": {"name": {"type": "string"}, "ip": {"type": "string"}},
            "required": ["name", "ip"],
        },
    ),
]


_OBSERVE = {"ping", "traceroute", "dig", "show_hosts", "show_links", "show_routes", "show_acl"}
for _t in TOOLS:  # documentation for the Environments page; models never see it
    _t.category = "observe" if _t.name in _OBSERVE else "act"

# Fixed drawing positions (x, y in [0, 1]) matching the ASCII map in SYSTEM_PROMPT.
_POS = {"R1": (0.12, 0.5), "R2": (0.42, 0.5), "R3": (0.78, 0.18), "R4": (0.78, 0.82)}
_SITE = {"R1": "siteA", "R2": "core", "R3": "siteB", "R4": "dc"}


def _cidr(s: str):
    if s == "any":
        return net("0.0.0.0/0")
    try:
        return net(s)
    except ValueError as e:
        raise ToolError(f"bad CIDR '{s}': {e}") from e


class NetopsEnvironment(Environment):
    name: ClassVar[str] = "netops"
    kind: ClassVar[str] = "sim"
    max_steps: ClassVar[int] = 25
    tiers: ClassVar[tuple[int, ...]] = (1, 2, 3)
    description: ClassVar[str] = (
        "Diagnose and repair a broken enterprise network (routes, links, ACLs, DNS) "
        "without collateral damage."
    )

    def __init__(self, task: Task) -> None:
        super().__init__(task)
        self._scn: F.Scenario | None = None
        self._net: Network | None = None
        self._changes: list[str] = []

    # ---- task generation -------------------------------------------------
    @classmethod
    def generate_task(cls, seed: int, tier: int) -> Task:
        scn = F.generate(seed, tier)
        return Task(
            id=f"netops-t{tier}-s{seed}",
            pack=cls.name,
            tier=tier,
            seed=seed,
            goal=scn.goal_text(),
            hidden={
                "faults": [f.description for f in scn.faults],
                "src_host": scn.src_host,
                "dst_host": scn.dst_host,
                "must_block": scn.must_block,
                "decoy_rules": scn.decoy_rules,
            },
        )

    # ---- lifecycle -------------------------------------------------------
    def reset(self) -> Observation:
        self._scn = F.generate(self.task.seed, self.task.tier)  # deterministic rebuild
        self._net = self._scn.network
        self._changes = []
        text = self._net.render_hosts() + "\n\n" + self._net.render_links()
        return Observation(text=text)

    def tools(self) -> list[ToolSpec]:
        return TOOLS

    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    @property
    def net(self) -> Network:
        assert self._net is not None, "call reset() first"
        return self._net

    def _router(self, name: Any) -> str:
        if name not in self.net.routers:
            raise ToolError(f"unknown router '{name}'. Routers: {sorted(self.net.routers)}")
        return name

    def step(self, call: ToolCall) -> Observation:
        n = self.net
        a = call.args
        t = call.name
        try:
            if t == "ping":
                tr = n.trace(a["src"], a["dst"])
                ip = f" ({tr.resolved_ip})" if tr.resolved_ip else ""
                return Observation(
                    text=f"ping {a['src']} -> {a['dst']}{ip}: "
                    + ("OK" if tr.ok else f"FAIL — {tr.reason}")
                )
            if t == "traceroute":
                tr = n.trace(a["src"], a["dst"])
                hops = " -> ".join(tr.path)
                tail = "" if tr.ok else f"\n  stopped: {tr.reason}"
                return Observation(text=f"traceroute {a['src']} -> {a['dst']}: {hops}{tail}")
            if t == "dig":
                ip = n.dns.get(a["name"])
                return Observation(text=f"{a['name']} -> {ip}" if ip else f"{a['name']}: NXDOMAIN")
            if t == "show_hosts":
                return Observation(text=n.render_hosts())
            if t == "show_links":
                return Observation(text=n.render_links())
            if t == "show_routes":
                return Observation(text=n.render_routes(self._router(a["router"])))
            if t == "show_acl":
                return Observation(text=n.render_acl(self._router(a["router"])))

            # ---- mutations
            if t == "set_route":
                r, nh = self._router(a["router"]), self._router(a["next_hop"])
                if r == nh:
                    raise ToolError("next_hop cannot be the router itself")
                n.routers[r].routes[_cidr(a["prefix"])] = nh
                self._changes.append(f"set_route {r} {a['prefix']} via {nh}")
                return Observation(text=f"{r}: route {a['prefix']} via {nh} installed")
            if t == "delete_route":
                r = self._router(a["router"])
                if n.routers[r].routes.pop(_cidr(a["prefix"]), None) is None:
                    raise ToolError(f"{r} has no static route for {a['prefix']}")
                self._changes.append(f"delete_route {r} {a['prefix']}")
                return Observation(text=f"{r}: route {a['prefix']} removed")
            if t == "remove_acl_rule":
                r = self._router(a["router"])
                rid = int(a["rule_id"])
                before = len(n.routers[r].acl)
                n.routers[r].acl = [x for x in n.routers[r].acl if x.id != rid]
                if len(n.routers[r].acl) == before:
                    raise ToolError(f"{r} has no ACL rule {rid}")
                self._changes.append(f"remove_acl_rule {r} {rid}")
                return Observation(text=f"{r}: ACL rule {rid} removed")
            if t == "add_acl_rule":
                from envforge.packs.netops.sim import AclRule

                r = self._router(a["router"])
                rule = AclRule(int(a["rule_id"]), a["action"], _cidr(a["src"]), _cidr(a["dst"]))
                if a["action"] not in ("permit", "deny"):
                    raise ToolError("action must be permit or deny")
                n.routers[r].acl.insert(0, rule)
                self._changes.append(f"add_acl_rule {r} {rule.render().strip()}")
                return Observation(text=f"{r}: rule inserted\n{rule.render()}")
            if t == "set_link":
                x, y = self._router(a["a"]), self._router(a["b"])
                try:
                    lk = n.link(x, y)
                except KeyError as e:
                    raise ToolError(str(e)) from e
                lk.up = a["state"] == "up"
                self._changes.append(f"set_link {x}-{y} {a['state']}")
                return Observation(text=f"link {x}-{y} is now {'up' if lk.up else 'down'}")
            if t == "set_dns":
                from envforge.packs.netops.sim import addr

                try:
                    ip = addr(a["ip"])
                except ValueError as e:
                    raise ToolError(f"bad IP '{a['ip']}'") from e
                n.dns[a["name"]] = ip
                self._changes.append(f"set_dns {a['name']} {ip}")
                return Observation(text=f"DNS: {a['name']} -> {ip}")
        except KeyError as e:
            raise ToolError(f"missing argument {e} for {t}") from e
        raise ToolError(f"unknown tool '{t}'")

    # ---- verification ----------------------------------------------------
    def verify(self) -> Verdict:
        scn, n = self._scn, self.net
        assert scn is not None
        ok, why = n.reachable(scn.src_host, scn.dst_host)
        collateral: list[str] = []
        for a, b in scn.must_reach:
            r, reason = n.reachable(a, b)
            if not r:
                collateral.append(f"broke {a}<->{b}: {reason}")
        for a, b in scn.must_block:
            fwd = n.trace(a, b)
            if fwd.ok:
                collateral.append(f"policy violated: {a} can now reach {b}")
        for router, rid in scn.decoy_rules:
            if not any(rule.id == rid for rule in n.routers[router].acl):
                collateral.append(f"removed security-policy ACL rule {rid} on {router}")

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
                "target_reachable": ok,
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
        return self.net.snapshot() if self._net else {}

    # ---- self-description (Environments page) ---------------------------
    @classmethod
    def describe(cls) -> EnvDescription:
        return (
            super()
            .describe()
            .model_copy(
                update={
                    "overview": (
                        "The agent is the on-call network engineer for a small enterprise WAN. "
                        "A user reports that one host cannot reach another. The agent must find "
                        "out why with read-only tools, repair it with the smallest change, and "
                        "leave every other working path and every security-policy rule intact."
                    ),
                    "world": [
                        "4 routers: three sites (R1, R3, R4) and a core (R2), joined by 4 links",
                        "static routing, longest-prefix match; one /24 LAN per site",
                        "per-router ACLs: top-down, first match wins, implicit permit",
                        "one DNS table; 2 hosts per site, names and addresses chosen per seed",
                        "the simulator walks packets hop by hop and reports where and why they stop",
                    ],
                    "observation": (
                        "Command-style text: ping/traceroute results with the failure reason, "
                        "routing tables, ACLs, link states, DNS answers."
                    ),
                    "action": (
                        "7 read-only tools to investigate and 6 tools that change routes, links, "
                        "ACL rules or DNS; several calls per turn; `finish` ends the episode."
                    ),
                    "tier_notes": {
                        1: "One fault: a missing route, a link down, an ACL deny, or stale DNS.",
                        2: "Two faults (adds wrong next-hop / routing loop) and a decoy "
                        "security-policy rule on the destination router that must survive.",
                        3: "Three faults including a missing return-path route, decoys, and "
                        "site-to-site targets.",
                    },
                    "reward_rules": [
                        RewardRule(
                            outcome="target reachable both ways, nothing else broken", reward=1.0
                        ),
                        RewardRule(
                            outcome="target fixed, but a bystander pair broke, a policy rule was "
                            "removed, or policy-blocked traffic now flows",
                            reward=0.3,
                        ),
                        RewardRule(outcome="target still unreachable", reward=0.0),
                    ],
                    "tables": [
                        InfoTable(
                            title="Fault catalogue",
                            columns=["fault", "symptom the agent sees", "typical fix"],
                            rows=[
                                ["route missing", "no route to … on Rx", "set_route"],
                                ["link down", "link Rx-Ry is down", "set_link up, or reroute"],
                                ["ACL deny", "denied by ACL on Rx (rule n)", "remove_acl_rule"],
                                ["stale DNS", "ping by name hits the wrong IP", "set_dns"],
                                ["wrong next-hop", "TTL exceeded (routing loop)", "set_route"],
                                ["missing return path", "forward works, reply dies", "set_route"],
                            ],
                        ),
                        InfoTable(
                            title="Addressing",
                            columns=["router", "role", "LAN"],
                            rows=[
                                [r, _SITE[r], SITES.get(r, "—")] for r in ("R1", "R2", "R3", "R4")
                            ],
                            note="Router-to-router links use /30s: "
                            + ", ".join(f"{a}–{b} {s}" for a, b, s in LINKS),
                        ),
                    ],
                }
            )
        )

    def diagram(self) -> Diagram | None:
        if self._net is None or self._scn is None:
            return None
        n, scn = self._net, self._scn
        fwd = n.trace(scn.src_host, scn.dst_host)
        back = n.trace(scn.dst_host, scn.src_host)
        policy_routers = {r for r, _ in scn.decoy_rules}
        changed = {c.split()[1] for c in self._changes if len(c.split()) > 1}

        def where_broken(t) -> str | None:
            # the last router a failed trace reached is where it died
            routers = [p for p in t.path if p in n.routers]
            return routers[-1] if not t.ok and routers else None

        broken = {r for r in (where_broken(fwd), where_broken(back)) if r}
        nodes: list[DiagramNode] = []
        for r, rt in n.routers.items():
            routes = "\n".join(f"  {p} via {nh}" for p, nh in sorted(rt.routes.items(), key=str))
            acl = "\n".join("  " + rule.render() for rule in rt.acl) or "  (none)"
            badges = (["policy"] if r in policy_routers else []) + (
                ["changed"] if r in changed else []
            )
            nodes.append(
                DiagramNode(
                    id=r,
                    label=r,
                    kind="router",
                    group=_SITE[r],
                    state="broken" if r in broken else "ok",
                    badges=badges,
                    detail=f"{r} ({_SITE[r]}, LAN {SITES.get(r, 'none')})\n"
                    f"static routes:\n{routes or '  (none)'}\nACL:\n{acl}",
                    x=_POS[r][0],
                    y=_POS[r][1],
                )
            )
        # hosts fan out beside their router
        per_router: dict[str, int] = {}
        for h in sorted(n.hosts.values(), key=lambda h: h.name):
            i = per_router.get(h.router, 0)
            per_router[h.router] = i + 1
            rx, ry = _POS[h.router]
            dx = -0.1 if rx < 0.5 else 0.12
            dns_ip = n.dns.get(h.name)
            dns_bad = dns_ip is not None and dns_ip != h.ip
            nodes.append(
                DiagramNode(
                    id=h.name,
                    label=h.name,
                    kind="host",
                    group=_SITE[h.router],
                    state="target" if h.name in (scn.src_host, scn.dst_host) else "ok",
                    badges=["dns"] if dns_bad else [],
                    detail=f"{h.name} {h.ip} behind {h.router}"
                    + (f"\nDNS says {dns_ip} (stale)" if dns_bad else ""),
                    x=rx + dx,
                    y=ry + (i - 0.5) * 0.16,
                )
            )
        on_path = set()
        for t in (fwd, back):
            hops = [p for p in t.path if p in n.routers]
            on_path |= {tuple(sorted(pair)) for pair in zip(hops, hops[1:], strict=False)}
        edges = [
            DiagramEdge(
                source=a,
                target=b,
                label=next(s for x, y, s in LINKS if {x, y} == {a, b}),
                state="broken" if not lk.up else "path" if (a, b) in on_path else "ok",
            )
            for (a, b), lk in n.links.items()
        ]
        edges += [DiagramEdge(source=h.router, target=h.name, state="ok") for h in n.hosts.values()]
        notes = [
            f"{scn.src_host} → {scn.dst_host}: " + ("OK" if fwd.ok else fwd.reason),
            f"{scn.dst_host} → {scn.src_host}: " + ("OK" if back.ok else back.reason),
        ]
        return Diagram(title="WAN topology", nodes=nodes, edges=edges, notes=notes)
