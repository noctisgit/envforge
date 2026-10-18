"""A small, deterministic network simulator.

No packets are sent. The "network" is data: routers, links, hosts, static
routes, ACLs and a DNS table. `trace()` walks a packet's path through that data
the way a real forwarding plane would, and reports *why* it failed if it does.

It is small on purpose. The point is not realism; it is a world with enough
structure that (a) faults have distinct, discoverable symptoms, (b) fixes can
break other things, and (c) success is checkable by code.

Topology (fixed; what varies per task is hosts, faults and decoys):

    siteA(R1) --- core(R2) --- siteB(R3)
                     |            |
                     +--- dc(R4)--+

R1: 10.1.0.0/24  R3: 10.3.0.0/24  R4: 10.4.0.0/24  (R2 has no hosts)
Router-to-router links use /30s in 10.0.x.0.
"""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass, field

IPNet = ipaddress.IPv4Network
IPAddr = ipaddress.IPv4Address


def net(s: str) -> IPNet:
    return ipaddress.ip_network(s, strict=False)


def addr(s: str) -> IPAddr:
    return ipaddress.ip_address(s)


@dataclass
class Host:
    name: str
    ip: IPAddr
    router: str


@dataclass
class Link:
    a: str
    b: str
    up: bool = True

    def other(self, r: str) -> str:
        return self.b if r == self.a else self.a

    def key(self) -> tuple[str, str]:
        return tuple(sorted((self.a, self.b)))  # type: ignore[return-value]


@dataclass
class AclRule:
    id: int
    action: str  # "permit" | "deny"
    src: IPNet  # 0.0.0.0/0 == any
    dst: IPNet
    note: str = ""

    def matches(self, s: IPAddr, d: IPAddr) -> bool:
        return s in self.src and d in self.dst

    def render(self) -> str:
        src = "any" if self.src.prefixlen == 0 else str(self.src)
        dst = "any" if self.dst.prefixlen == 0 else str(self.dst)
        note = f"   ! {self.note}" if self.note else ""
        return f"{self.id:>4}  {self.action:<6} {src:<18} -> {dst:<18}{note}"


@dataclass
class Router:
    name: str
    connected: dict[IPNet, str] = field(default_factory=dict)  # subnet -> iface name
    routes: dict[IPNet, str] = field(default_factory=dict)  # prefix -> next-hop router
    acl: list[AclRule] = field(default_factory=list)

    def lookup(self, dst: IPAddr) -> str | None:
        """Longest-prefix match. Returns next-hop router name, 'direct', or None."""
        for subnet in self.connected:
            if dst in subnet:
                return "direct"
        best: tuple[int, str] | None = None
        for prefix, nh in self.routes.items():
            if dst in prefix and (best is None or prefix.prefixlen > best[0]):
                best = (prefix.prefixlen, nh)
        return best[1] if best else None

    def acl_decision(self, s: IPAddr, d: IPAddr) -> AclRule | None:
        """First matching rule, or None (implicit permit)."""
        for rule in self.acl:
            if rule.matches(s, d):
                return rule
        return None


@dataclass
class TraceResult:
    ok: bool
    path: list[str]
    reason: str  # human-readable failure reason, "" when ok
    resolved_ip: IPAddr | None = None


class Network:
    MAX_HOPS = 12

    def __init__(self) -> None:
        self.routers: dict[str, Router] = {}
        self.links: dict[tuple[str, str], Link] = {}
        self.hosts: dict[str, Host] = {}
        self.dns: dict[str, IPAddr] = {}

    # ---- construction ----------------------------------------------------
    def add_router(self, name: str) -> Router:
        r = Router(name)
        self.routers[name] = r
        return r

    def add_link(self, a: str, b: str, subnet: str) -> None:
        self.links[tuple(sorted((a, b)))] = Link(a, b)  # type: ignore[arg-type]
        self.routers[a].connected[net(subnet)] = f"to-{b}"
        self.routers[b].connected[net(subnet)] = f"to-{a}"

    def add_lan(self, router: str, subnet: str) -> None:
        self.routers[router].connected[net(subnet)] = "lan"

    def add_host(self, name: str, ip: str, router: str) -> None:
        self.hosts[name] = Host(name, addr(ip), router)
        self.dns[name] = addr(ip)

    def link(self, a: str, b: str) -> Link:
        key = tuple(sorted((a, b)))
        if key not in self.links:
            raise KeyError(f"no link between {a} and {b}")
        return self.links[key]  # type: ignore[index]

    def resolve(self, name_or_ip: str) -> IPAddr | None:
        try:
            return addr(name_or_ip)
        except ValueError:
            return self.dns.get(name_or_ip)

    def host_by_ip(self, ip: IPAddr) -> Host | None:
        for h in self.hosts.values():
            if h.ip == ip:
                return h
        return None

    def router_for_ip(self, ip: IPAddr) -> str | None:
        for r in self.routers.values():
            for subnet, iface in r.connected.items():
                if iface == "lan" and ip in subnet:
                    return r.name
        return None

    # ---- forwarding ------------------------------------------------------
    def trace(self, src_host: str, dst: str) -> TraceResult:
        """Walk a packet from src_host to dst (name or IP). One direction only."""
        if src_host not in self.hosts:
            return TraceResult(False, [], f"unknown source host '{src_host}'")
        src = self.hosts[src_host]
        dst_ip = self.resolve(dst)
        if dst_ip is None:
            return TraceResult(False, [], f"DNS: cannot resolve '{dst}'")

        path = [src.name]
        cur = src.router
        hops = 0
        while True:
            if hops > self.MAX_HOPS:
                return TraceResult(False, path, "TTL exceeded (routing loop)", dst_ip)
            r = self.routers[cur]
            path.append(cur)
            rule = r.acl_decision(src.ip, dst_ip)
            if rule and rule.action == "deny":
                return TraceResult(False, path, f"denied by ACL on {cur} (rule {rule.id})", dst_ip)
            nh = r.lookup(dst_ip)
            if nh is None:
                return TraceResult(False, path, f"no route to {dst_ip} on {cur}", dst_ip)
            if nh == "direct":
                if self.router_for_ip(dst_ip) == cur and self.host_by_ip(dst_ip):
                    path.append(str(dst_ip))
                    return TraceResult(True, path, "", dst_ip)
                return TraceResult(
                    False,
                    path,
                    f"{dst_ip} is in a directly connected subnet on {cur} but no host answers",
                    dst_ip,
                )
            if nh not in self.routers:
                return TraceResult(
                    False, path, f"route on {cur} points to unknown next-hop '{nh}'", dst_ip
                )
            try:
                lk = self.link(cur, nh)
            except KeyError:
                return TraceResult(
                    False, path, f"route on {cur} points to {nh} but there is no link", dst_ip
                )
            if not lk.up:
                return TraceResult(False, path, f"link {cur}-{nh} is down", dst_ip)
            cur = nh
            hops += 1

    def reachable(self, a: str, b: str) -> tuple[bool, str]:
        """Bidirectional connectivity between two hosts by name."""
        fwd = self.trace(a, b)
        if not fwd.ok:
            return False, f"{a}->{b}: {fwd.reason}"
        back = self.trace(b, a)
        if not back.ok:
            return False, f"{b}->{a}: {back.reason}"
        return True, ""

    # ---- rendering -------------------------------------------------------
    def render_routes(self, router: str) -> str:
        r = self.routers[router]
        lines = [f"{router} routing table"]
        for subnet, iface in sorted(r.connected.items(), key=lambda kv: str(kv[0])):
            lines.append(f"  C  {str(subnet):<18} directly connected ({iface})")
        for prefix, nh in sorted(r.routes.items(), key=lambda kv: str(kv[0])):
            lines.append(f"  S  {str(prefix):<18} via {nh}")
        return "\n".join(lines)

    def render_acl(self, router: str) -> str:
        r = self.routers[router]
        if not r.acl:
            return f"{router} has no ACL rules (implicit permit)"
        lines = [f"{router} ACL (evaluated top-down, first match wins, implicit permit at end)"]
        lines += [rule.render() for rule in r.acl]
        return "\n".join(lines)

    def render_links(self) -> str:
        lines = ["links"]
        for lk in sorted(self.links.values(), key=lambda lk_: lk_.key()):
            lines.append(f"  {lk.a}-{lk.b}  {'up' if lk.up else 'DOWN'}")
        return "\n".join(lines)

    def render_hosts(self) -> str:
        lines = ["hosts"]
        for h in sorted(self.hosts.values(), key=lambda h: h.name):
            lines.append(f"  {h.name:<16} {str(h.ip):<14} on {h.router}")
        return "\n".join(lines)

    def snapshot(self) -> dict:
        return {
            "links": {f"{a}-{b}": lk.up for (a, b), lk in self.links.items()},
            "routes": {
                r: {str(p): nh for p, nh in rt.routes.items()} for r, rt in self.routers.items()
            },
            "acl": {r: [rule.render() for rule in rt.acl] for r, rt in self.routers.items()},
            "dns": {k: str(v) for k, v in self.dns.items()},
        }


# ---- the base topology ------------------------------------------------------

SITES = {"R1": "10.1.0.0/24", "R3": "10.3.0.0/24", "R4": "10.4.0.0/24"}
LINKS = [
    ("R1", "R2", "10.0.12.0/30"),
    ("R2", "R3", "10.0.23.0/30"),
    ("R2", "R4", "10.0.24.0/30"),
    ("R3", "R4", "10.0.34.0/30"),
]


def _shortest_paths() -> dict[str, dict[str, str]]:
    """next_hop[router][dest_router] via BFS on the base links."""
    adj: dict[str, set[str]] = {}
    for a, b, _ in LINKS:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    nh: dict[str, dict[str, str]] = {}
    for start in adj:
        prev: dict[str, str | None] = {start: None}
        queue = [start]
        for cur in queue:
            for n in sorted(adj[cur]):
                if n not in prev:
                    prev[n] = cur
                    queue.append(n)
        nh[start] = {}
        for dest in adj:
            if dest == start:
                continue
            node = dest
            while prev[node] != start:
                node = prev[node]  # type: ignore[assignment]
            nh[start][dest] = node
    return nh


def build_base_network() -> Network:
    """A fully working network. Faults are injected on top of this."""
    n = Network()
    for r in ("R1", "R2", "R3", "R4"):
        n.add_router(r)
    for a, b, subnet in LINKS:
        n.add_link(a, b, subnet)
    for r, subnet in SITES.items():
        n.add_lan(r, subnet)
    nh = _shortest_paths()
    for r in n.routers:
        for site_router, subnet in SITES.items():
            if site_router != r:
                n.routers[r].routes[net(subnet)] = nh[r][site_router]
    return n
