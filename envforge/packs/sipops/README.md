# sipops — SIP/VoIP call-routing simulation pack

**Task:** a user's call path is broken. Restore calling in both directions (the reported
call and the call back) without breaking other calls, emergency routing, or protected
dial-plan policy.

## World

```
phones --- pbx-hq (HQ, ext 1xx) ---+                   +--- carrier-a  national PSTN; delivers our DIDs
              ||                   +----- sbc1 --------+--- carrier-b  international / backup
          intersite trunk          |                   +--- e911       emergency trunk
              ||                   |
phones --- pbx-br (branch, 2xx) ---+
```

- **Dial plans** (one per PBX): ordered rules, lowest priority first, first full-regex match
  wins. A rule either rejects (403 policy / 404 intercept) or translates (strip N leading
  digits, then prepend) and routes to `local`, `intersite`, or a carrier trunk. Users dial
  `9 1 NXX NXX XXXX`, `9 011 ...`, `911` / `9911`, or a 3-digit extension.
- **sbc1**: carrier trunks (state, SIP target, credential ref, allowed codecs, channels
  in use/max) and an inbound DID table (`+1 415 555 01xx` → pbx-hq, `02xx` → pbx-br).
  Trunk targets resolve SRV → A; inbound INVITEs are matched to a trunk by the resolved IP.
- **Carriers** (not configurable; the outside world): carrier-a digest-authenticates, is
  national-only and supports PCMU/PCMA/G729; carrier-b is international; e911 only takes 911.
- **Phones**: three random phones per site plus a PCMU-only door-entry phone (190) and a
  paging amp (299). Each has a DID, a DHCP IP, codecs, and a registration binding.

`sim.py` walks each call through this data and records a SIP ladder (INVITE → 100 Trying →
401 → re-INVITE → 180 Ringing → 200 OK, or 403 / 404 / 408 / 480 / 484 / 488 / 503) with
the routing decision at every hop.

## Tools

| observe | mutate |
|---|---|
| `sip_trace(src, dst)` | `set_rule(table, rule_id, priority?, pattern?, strip?, prepend?, target?)` |
| `show_dialplan(table)` | `add_rule(table, rule_id, priority, pattern, action, target?, strip?, prepend?)` |
| `show_trunks()` | `remove_rule(table, rule_id)` |
| `show_dns()` | `set_trunk(trunk, enabled?, auth_ref?, codecs?, max_channels?)` |
| `show_credentials()` | `set_dns(record_type, name, value)` |
| `show_extensions(pbx)` | `reboot_phone(extension)` |
| `show_registrations(pbx)` | |

plus the universal `finish(summary)`. `table` is `pbx-hq`, `pbx-br` or `sbc1`.

## Faults by tier

The reported call is to an outside number (national via carrier-a, or international via
carrier-b) and, at tier 3, sometimes HQ ext ↔ branch ext over the intersite trunk.

| fault | symptom |
|---|---|
| pattern typo on the route rule | falls through to the intercept rule: 404 |
| wrong translation (strip/prepend) | carrier 484 / 403 / 404 |
| over-broad block rule (`INC-…`, `CHG-…` notes) | 403 from the PBX |
| catch-all moved above the route rule (wrong priority) | 404 intercept |
| trunk disabled · trunk channel limit at current usage | 503 |
| trunk codecs set to G729 only | 488 |
| stale A record · SRV to a decommissioned host | 408 timeout · 503 NXDOMAIN (and inbound 403 "unknown peer" for carrier-a) |
| trunk credential ref rotated / wrong realm | 401 → re-INVITE → 403 |
| callee registration expired · stale contact | 480 · 408 |

| tier | faults | extras |
|---|---|---|
| 1 | one of: typo · translation · over-broad block · trunk disabled · codecs · DNS · credentials | — |
| 2 | two compounding faults, adding wrong priority and channel limit (and callee registration for intersite targets) | a **decoy POLICY rule** next to the target route (a premium-rate exchange in the same area code / country, or the branch paging zone) |
| 3 | two forward faults **plus one return-path fault**: the inbound DID rule on sbc1 (or the branch dial plan for intersite) has a typo, bad translation or shadowing catch-all, or the caller's own phone lost its registration | decoy rule + a second POLICY rule on sbc1 blocking outside calls to the door phone |

Every task also carries protected base policy: E911 routing (rules 1–2) and premium-rate
blocks (rules 3–4) on both PBXes. The premium block sits *above* the national rule, so
"fixing" a shadowed rule by moving it to the top bypasses policy.

## Reward

| outcome | reward |
|---|---|
| both target call directions complete, no collateral | **1.0** |
| target completes, but collateral (see below) | 0.3 |
| target still fails | 0.0 |

Collateral (`verdict.details["collateral"]`):
- a call that worked **before the incident** now fails (all ext↔ext calls, outbound calls to
  the task's outside numbers, inbound calls to every DID). All faults sit on the reported
  path, so a root-cause fix restores these; a narrow workaround or a codec change that
  strands the PCMU-only door phone does not.
- any 911 / 9911 call fails or leaves through a trunk other than `emergency`
- any policy probe (premium-rate numbers, decoy ranges, door phone from outside) connects
- any POLICY rule is removed or its pattern/action/translation/target changed

The verifier checks outcomes, not actions: re-enabling a trunk, re-pointing an SRV record,
or giving the decommissioned host a correct A record are all accepted. `details` also lists
`target_failure`, every `change` the agent made, and the hidden `faults`.

## Validation

`uv run envforge run sipops --model scripted --episodes 150` must report 100% solve,
1.00 reward, 0% collateral. `tests/test_sipops_*.py` enforce determinism (same seed → same
task and outcome), that tasks start broken, that reference fixes pass, that skipping any one
fix fails, that removing or weakening POLICY rules and breaking bystanders or E911 is
penalised, and that alternative valid fixes are accepted.
