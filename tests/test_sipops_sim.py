"""Simulator-level tests: the world behaves like a SIP voice platform."""

from envforge.packs.sipops.sim import Extension, Rule, build_base_platform

CODECS = ["G722", "PCMU", "PCMA"]


def _platform():
    p = build_base_platform()
    p.add_extension(
        Extension("101", "reception", "pbx-hq", "Yealink", "10.10.1.21", CODECS, "+14155550101")
    )
    p.add_extension(
        Extension("201", "front", "pbx-br", "Yealink", "10.20.1.21", CODECS, "+14155550201")
    )
    p.pstn["+12125550147"] = "Acme Freight"
    p.pstn["+442079460123"] = "Thames Imports"
    return p


def _code(res) -> int:
    """The SIP status code the caller finally received."""
    return int(res.reason.split()[0])


def test_base_platform_completes_every_call():
    p = _platform()
    calls = [
        ("101", "201"),  # intersite
        ("201", "101"),
        ("101", "190"),  # local, PCMU-only door phone
        ("101", "912125550147"),  # national via carrier-a
        ("201", "9011442079460123"),  # international via carrier-b
        ("+12125550147", "+14155550101"),  # inbound DID
        ("+12125550147", "+14155550201"),
        ("101", "911"),
        ("201", "9911"),
    ]
    for src, dst in calls:
        res = p.call(src, dst)
        assert res.ok, (src, dst, res.reason)
    assert p.call("101", "911").egress == "emergency"


def test_premium_numbers_are_rejected_by_policy():
    p = _platform()
    res = p.call("101", "919005550199")
    assert not res.ok and _code(res) == 403
    # ...and they would connect without the rule: that is why the rule matters
    p.tables["pbx-hq"] = [r for r in p.tables["pbx-hq"] if r.id != 3]
    assert p.call("101", "919005550199").ok


def test_first_match_wins_by_priority():
    p = _platform()
    p.tables["pbx-hq"].append(Rule(20, 45, r"91212\d{7}", action="reject"))
    assert _code(p.call("101", "912125550147")) == 403
    p.tables["pbx-hq"][-1].priority = 55  # now below the national rule
    assert p.call("101", "912125550147").ok


def test_pattern_typo_falls_through_to_intercept():
    p = _platform()
    p.rule("pbx-hq", 7).pattern = r"91[2-9]\d{8}"
    res = p.call("101", "912125550147")
    assert _code(res) == 404 and "rule 9" in res.reason


def test_bad_translation_is_rejected_by_carrier():
    p = _platform()
    p.rule("pbx-hq", 7).prepend = ""  # sends 12125550147, not E.164
    assert _code(p.call("101", "912125550147")) == 484


def test_disabled_trunk_and_channel_limit_give_503():
    p = _platform()
    p.trunks["carrier-a"].enabled = False
    assert _code(p.call("101", "912125550147")) == 503
    p.trunks["carrier-a"].enabled = True
    p.trunks["carrier-a"].max_channels = p.trunks["carrier-a"].active_calls
    res = p.call("101", "912125550147")
    assert _code(res) == 503 and "channels" in res.reason


def test_codec_mismatch_gives_488():
    p = _platform()
    p.trunks["carrier-a"].codecs = ["G729"]
    assert _code(p.call("101", "912125550147")) == 488
    # dropping PCMU leaves PCMA for most phones but strands the PCMU-only door phone inbound
    p.trunks["carrier-a"].codecs = ["PCMA"]
    assert p.call("+12125550147", "+14155550101").ok
    assert _code(p.call("+12125550147", "+14155550190")) == 488


def test_wrong_credential_gives_401_then_403():
    p = _platform()
    p.trunks["carrier-a"].auth_ref = "cred-carrier-a-2023"  # rotated
    res = p.call("101", "912125550147")
    assert _code(res) == 403
    assert any("401 Unauthorized" in line for line in res.ladder)


def test_stale_a_record_times_out_and_breaks_inbound_peer_match():
    p = _platform()
    p.dns_a["edge1.carrier-a.net"] = "198.51.100.97"
    assert _code(p.call("101", "912125550147")) == 408
    inbound = p.call("+12125550147", "+14155550101")
    assert _code(inbound) == 403 and "unknown peer" in inbound.reason


def test_srv_to_missing_host_gives_503():
    p = _platform()
    p.dns_srv["_sip._udp.carrier-b.net"] = ("sbc-old.carrier-b.net", 5060)
    res = p.call("101", "9011442079460123")
    assert _code(res) == 503 and "NXDOMAIN" in res.reason


def test_registration_problems():
    p = _platform()
    p.registrations["201"].expires = 0
    assert _code(p.call("101", "201")) == 480
    p.registrations["201"].expires = 600
    p.registrations["201"].contact = "10.20.1.170"
    res = p.call("101", "201")
    assert _code(res) == 408 and "pbx-br" in res.reason


def test_inbound_translation_error_gives_404_on_pbx():
    p = _platform()
    p.rule("sbc1", 1).strip = 8
    res = p.call("+12125550147", "+14155550101")
    assert _code(res) == 404 and "pbx-hq" in res.reason


def test_national_carrier_refuses_international():
    p = _platform()
    p.rule("pbx-hq", 8).target = "carrier-a"
    assert _code(p.call("101", "9011442079460123")) == 403


def test_reroute_around_dead_trunk_is_accepted():
    """Re-enabling the trunk is not the only repair: the backup carrier takes national calls too."""
    p = _platform()
    p.trunks["carrier-a"].enabled = False
    p.rule("pbx-hq", 7).target = "carrier-b"
    res = p.call("101", "912125550147")
    assert res.ok and res.egress == "carrier-b"
