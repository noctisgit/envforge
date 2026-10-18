"""Fault injection: the synthetic task generator for netops.

Each fault knows how to break the network, what a correct fix looks like
(for the scripted model and the tests), and a hidden one-line description
(for debugging and for `envforge show`). The generator composes faults by
tier and adds *decoys* — security-policy ACL rules that must survive.

This file is the "synthetic data pipeline" piece of the pack: the same
seed always yields the same task.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from envforge.core.env import ToolCall
from envforge.packs.netops.sim import SITES, AclRule, Network, net

# host name pools per site router
HOST_POOLS = {
    "R1": ["ws-a1", "ws-a2", "printer-a1", "kiosk-a1"],
    "R3": ["app-b1", "app-b2", "ws-b1", "nas-b1"],
    "R4": ["db-d1", "api-d1", "auth-d1", "backup-d1"],
}


def _tc(tool: str, **args) -> ToolCall:
    """Build a reference ToolCall (ids only need to be unique within an episode)."""
    sig = "_".join(f"{k}-{v}" for k, v in sorted(args.items()))
    return ToolCall(id=f"ref_{tool}_{sig}"[:64], name=tool, args=args)


@dataclass
class Fault:
    kind: str
    description: str  # hidden
    fix: list[ToolCall]

    def apply(self, n: Network) -> None:  # overridden per kind
        raise NotImplementedError


@dataclass
class RouteMissing(Fault):
    router: str = ""
    prefix: str = ""
    next_hop: str = ""

    def apply(self, n: Network) -> None:
        n.routers[self.router].routes.pop(net(self.prefix), None)


@dataclass
class RouteWrong(Fault):
    router: str = ""
    prefix: str = ""
    good_hop: str = ""
    bad_hop: str = ""

    def apply(self, n: Network) -> None:
        n.routers[self.router].routes[net(self.prefix)] = self.bad_hop


@dataclass
class LinkDown(Fault):
    a: str = ""
    b: str = ""

    def apply(self, n: Network) -> None:
        n.link(self.a, self.b).up = False


@dataclass
class AclDeny(Fault):
    router: str = ""
    rule_id: int = 0
    src: str = ""
    dst: str = ""
    note: str = ""

    def apply(self, n: Network) -> None:
        n.routers[self.router].acl.insert(
            0, AclRule(self.rule_id, "deny", net(self.src), net(self.dst), self.note)
        )


@dataclass
class DnsStale(Fault):
    name: str = ""
    good_ip: str = ""
    bad_ip: str = ""

    def apply(self, n: Network) -> None:
        n.dns[self.name] = net(self.bad_ip).network_address


@dataclass
class Scenario:
    seed: int
    tier: int
    network: Network
    src_host: str
    dst_host: str
    faults: list[Fault]
    must_reach: list[tuple[str, str]]  # pairs that must still work at the end
    must_block: list[tuple[str, str]]  # pairs that policy must keep blocked
    decoy_rules: list[tuple[str, int]]  # (router, rule_id) that must survive

    def goal_text(self) -> str:
        return (
            f"Users on {self.src_host} report they cannot reach {self.dst_host}. "
            f"Diagnose and restore bidirectional connectivity between {self.src_host} and "
            f"{self.dst_host}. Do not break other working services, and do not remove ACL "
            f"rules marked as security policy. Call finish when done."
        )


def _path_routers(n: Network, src_host: str, dst_host: str) -> list[str]:
    """Routers on the working path src->dst (base network, no faults)."""
    tr = n.trace(src_host, dst_host)
    return [p for p in tr.path if p in n.routers]


def generate(seed: int, tier: int) -> Scenario:
    rng = random.Random(seed)
    from envforge.packs.netops.sim import build_base_network

    n = build_base_network()

    # hosts: 2 per site, fixed order so seeds stay stable
    for router, pool in HOST_POOLS.items():
        names = rng.sample(pool, 2)
        base = SITES[router].split("/")[0].rsplit(".", 1)[0]
        for i, name in enumerate(names):
            n.add_host(name, f"{base}.{10 + i}", router)

    # target pair: a site host -> a DC host (or site-to-site at tier 3)
    site_routers = ["R1", "R3"]
    src_router = rng.choice(site_routers)
    dst_router = "R4" if tier < 3 or rng.random() < 0.6 else ("R3" if src_router == "R1" else "R1")
    src_host = rng.choice([h.name for h in n.hosts.values() if h.router == src_router])
    dst_host = rng.choice([h.name for h in n.hosts.values() if h.router == dst_router])
    dst_subnet = SITES[dst_router]
    src_subnet = SITES[src_router]

    fwd_path = _path_routers(n, src_host, dst_host)
    rev_path = _path_routers(n, dst_host, src_host)

    # ---- decoys: security policy rules that block OTHER traffic and must survive
    decoy_rules: list[tuple[str, int]] = []
    must_block: list[tuple[str, str]] = []
    if tier >= 2:
        # block the *other* site host from reaching the DC target subnet
        other_site = "R3" if src_router == "R1" else "R1"
        blocked_host = rng.choice([h.name for h in n.hosts.values() if h.router == other_site])
        if dst_router != other_site:
            # Edge filtering on the destination router: every path to dst crosses it,
            # so the policy holds regardless of how the agent repairs routing.
            policy_router = dst_router
            rule_id = 10
            n.routers[policy_router].acl.append(
                AclRule(
                    rule_id,
                    "deny",
                    net(f"{n.hosts[blocked_host].ip}/32"),
                    net(dst_subnet),
                    note="SECURITY POLICY SEC-0071 — do not remove",
                )
            )
            decoy_rules.append((policy_router, rule_id))
            must_block.append((blocked_host, dst_host))

    # ---- faults
    n_faults = {1: 1, 2: 2, 3: 3}[tier]
    kinds_by_tier = {
        1: ["route_missing", "link_down", "acl_deny", "dns_stale"],
        2: ["route_missing", "route_wrong", "link_down", "acl_deny", "dns_stale"],
        3: [
            "route_missing",
            "route_wrong",
            "link_down",
            "acl_deny",
            "dns_stale",
            "return_route_missing",
        ],
    }
    kinds = rng.sample(kinds_by_tier[tier], n_faults)
    # route faults are applied in a fixed order so "wrong" never silently overrides "missing"
    order = {"route_wrong": 0, "route_missing": 1, "return_route_missing": 2}
    kinds.sort(key=lambda k: order.get(k, 9))
    faults: list[Fault] = []
    used_rule_ids = {10}
    touched_routes: set[tuple[str, str]] = set()  # (router, prefix) already faulted

    for kind in kinds:
        if kind == "route_missing":
            candidates = [r for r in fwd_path[:-1] if (r, dst_subnet) not in touched_routes]
            router = rng.choice(candidates or fwd_path[:-1])
            touched_routes.add((router, dst_subnet))
            nh = n.routers[router].routes[net(dst_subnet)]
            faults.append(
                RouteMissing(
                    "route_missing",
                    f"{router} lost its route to {dst_subnet} (was via {nh})",
                    [_tc("set_route", router=router, prefix=dst_subnet, next_hop=nh)],
                    router=router,
                    prefix=dst_subnet,
                    next_hop=nh,
                )
            )
        elif kind == "return_route_missing":
            router = rng.choice(rev_path[:-1])
            nh = n.routers[router].routes[net(src_subnet)]
            faults.append(
                RouteMissing(
                    "return_route_missing",
                    f"{router} lost its RETURN route to {src_subnet} (was via {nh})",
                    [_tc("set_route", router=router, prefix=src_subnet, next_hop=nh)],
                    router=router,
                    prefix=src_subnet,
                    next_hop=nh,
                )
            )
        elif kind == "route_wrong":
            # pick a transit router and point its route back the way it came -> loop
            if len(fwd_path) >= 3:
                idx = rng.randrange(1, len(fwd_path) - 1)
                router, prev_hop = fwd_path[idx], fwd_path[idx - 1]
            else:
                router, prev_hop = fwd_path[0], None
            touched_routes.add((router, dst_subnet))
            good = n.routers[router].routes[net(dst_subnet)]
            bad = prev_hop or next(r for r in n.routers if r not in (router, good))
            faults.append(
                RouteWrong(
                    "route_wrong",
                    f"{router} route to {dst_subnet} points to {bad} instead of {good}",
                    [_tc("set_route", router=router, prefix=dst_subnet, next_hop=good)],
                    router=router,
                    prefix=dst_subnet,
                    good_hop=good,
                    bad_hop=bad,
                )
            )
        elif kind == "link_down":
            a, b = rng.choice(list(zip(fwd_path, fwd_path[1:], strict=False)))
            faults.append(
                LinkDown(
                    "link_down",
                    f"link {a}-{b} is administratively down",
                    [_tc("set_link", a=a, b=b, state="up")],
                    a=a,
                    b=b,
                )
            )
        elif kind == "acl_deny":
            router = rng.choice(fwd_path)
            rule_id = rng.choice([r for r in (20, 30, 40, 50) if r not in used_rule_ids])
            used_rule_ids.add(rule_id)
            src_ip = f"{n.hosts[src_host].ip}/32"
            note = rng.choice(
                ["temporary block CHG-4412 (expired)", "added during incident INC-2291", ""]
            )
            faults.append(
                AclDeny(
                    "acl_deny",
                    f"{router} ACL rule {rule_id} denies {src_ip} -> {dst_subnet}",
                    [_tc("remove_acl_rule", router=router, rule_id=rule_id)],
                    router=router,
                    rule_id=rule_id,
                    src=src_ip,
                    dst=dst_subnet,
                    note=note,
                )
            )
        elif kind == "dns_stale":
            good_ip = str(n.hosts[dst_host].ip)
            base = dst_subnet.split("/")[0].rsplit(".", 1)[0]
            bad_ip = f"{base}.{rng.randint(200, 250)}"
            faults.append(
                DnsStale(
                    "dns_stale",
                    f"DNS record {dst_host} points to {bad_ip} instead of {good_ip}",
                    [_tc("set_dns", name=dst_host, ip=good_ip)],
                    name=dst_host,
                    good_ip=good_ip,
                    bad_ip=bad_ip,
                )
            )

    for f in faults:
        f.apply(n)

    # ---- must_reach: every pair that still works *after* the faults must keep
    # working at the end. "Don't break what wasn't broken."
    must_reach: list[tuple[str, str]] = []
    hosts = sorted(n.hosts)
    for a in hosts:
        for b in hosts:
            if a < b and {a, b} != {src_host, dst_host}:
                ok, _ = n.reachable(a, b)
                if ok:
                    must_reach.append((a, b))

    return Scenario(
        seed=seed,
        tier=tier,
        network=n,
        src_host=src_host,
        dst_host=dst_host,
        faults=faults,
        must_reach=must_reach,
        must_block=must_block,
        decoy_rules=decoy_rules,
    )
