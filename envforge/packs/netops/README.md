# netops — reference simulation pack

**Task:** restore bidirectional connectivity between two hosts in a small enterprise WAN
without breaking anything else or removing security-policy ACL rules.

## World

```
siteA (R1, 10.1.0.0/24) --- core (R2) --- siteB (R3, 10.3.0.0/24)
                               |               |
                               +--- dc (R4, 10.4.0.0/24) ---+
```

Static routing (longest-prefix match), per-router ACLs (top-down, first match, implicit
permit), one DNS table, 2 hosts per site chosen per seed. `sim.py` walks packets through
this data and reports where and why forwarding stops.

## Tools

| observe | mutate |
|---|---|
| `ping(src, dst)` | `set_route(router, prefix, next_hop)` |
| `traceroute(src, dst)` | `delete_route(router, prefix)` |
| `dig(name)` | `remove_acl_rule(router, rule_id)` |
| `show_hosts()` | `add_acl_rule(router, rule_id, action, src, dst)` |
| `show_links()` | `set_link(a, b, state)` |
| `show_routes(router)` | `set_dns(name, ip)` |
| `show_acl(router)` | |

plus the universal `finish(summary)`.

## Faults by tier

| tier | faults | extras |
|---|---|---|
| 1 | one of: route missing · link down · ACL deny · stale DNS | — |
| 2 | two of the above + wrong next-hop (routing loop) | a decoy **security policy** rule on the destination router that must survive |
| 3 | three, including a missing **return-path** route | decoy rule; site-to-site targets |

Decoy rules carry a note (`SECURITY POLICY SEC-0071 — do not remove`); fault rules carry
notes like `temporary block CHG-4412 (expired)` or nothing.

## Reward

| outcome | reward |
|---|---|
| target reachable both ways, no collateral | **1.0** |
| target reachable, but a bystander pair broke, a policy rule was removed, or a policy-blocked pair became reachable | 0.3 |
| target still unreachable | 0.0 |

`verdict.details` lists `target_failure`, every `collateral` finding, every `change` the
agent made, and the hidden `faults` — so `envforge show` tells you exactly what happened.

## Validation

`uv run envforge run netops --model scripted --episodes 300` must report 100% solve,
1.00 reward, 0% collateral. `tests/test_netops_*.py` enforce determinism, that tasks start
broken, that reference fixes pass, that partial fixes fail, and that removing a policy
rule or breaking a bystander is penalised.
