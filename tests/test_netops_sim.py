"""Simulator-level tests: the world behaves like a network."""

from envforge.packs.netops.sim import AclRule, build_base_network, net


def _hosts(n):
    n.add_host("a1", "10.1.0.10", "R1")
    n.add_host("b1", "10.3.0.10", "R3")
    n.add_host("d1", "10.4.0.10", "R4")
    return n


def test_base_network_is_fully_connected():
    n = _hosts(build_base_network())
    for a in ("a1", "b1", "d1"):
        for b in ("a1", "b1", "d1"):
            if a != b:
                ok, why = n.reachable(a, b)
                assert ok, why


def test_trace_reports_missing_route():
    n = _hosts(build_base_network())
    n.routers["R2"].routes.pop(net("10.4.0.0/24"))
    tr = n.trace("a1", "d1")
    assert not tr.ok and "no route" in tr.reason and tr.path[-1] == "R2"


def test_trace_reports_link_down():
    n = _hosts(build_base_network())
    n.link("R2", "R4").up = False
    tr = n.trace("a1", "d1")
    assert not tr.ok and "down" in tr.reason


def test_trace_detects_routing_loop():
    n = _hosts(build_base_network())
    n.routers["R2"].routes[net("10.4.0.0/24")] = "R1"  # points back
    tr = n.trace("a1", "d1")
    assert not tr.ok and "loop" in tr.reason


def test_acl_deny_first_match_wins():
    n = _hosts(build_base_network())
    n.routers["R4"].acl.append(AclRule(10, "deny", net("10.1.0.10/32"), net("10.4.0.0/24")))
    assert not n.trace("a1", "d1").ok
    assert n.trace("b1", "d1").ok  # other sources unaffected
    n.routers["R4"].acl.insert(0, AclRule(5, "permit", net("10.1.0.10/32"), net("10.4.0.0/24")))
    assert n.trace("a1", "d1").ok  # permit above the deny wins


def test_dns_stale_resolves_to_wrong_ip():
    n = _hosts(build_base_network())
    n.dns["d1"] = net("10.4.0.200/32").network_address
    tr = n.trace("a1", "d1")
    assert not tr.ok and "no host answers" in tr.reason
    assert str(tr.resolved_ip) == "10.4.0.200"


def test_alternate_path_fix_is_accepted():
    """Bringing the link back is not the only valid repair: rerouting works too."""
    n = _hosts(build_base_network())
    n.link("R2", "R4").up = False
    n.routers["R2"].routes[net("10.4.0.0/24")] = "R3"  # go around via siteB
    assert n.trace("a1", "d1").ok
