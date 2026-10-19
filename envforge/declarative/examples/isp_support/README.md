# isp_support — a declarative example environment

A fictional home-internet provider, **Lumen Broadband**, and its support desk. The whole
environment is data: `spec.json` (an `EnvSpec`) and `tasks.jsonl` (31 `TaskRecord`s). No
Python is needed to run it; `generate.py` is only how the files were written.

```bash
uv run envforge run envforge/declarative/examples/isp_support --model scripted --episodes 60
uv run envforge env validate envforge/declarative/examples/isp_support   # once env_app is mounted
uv run envforge env preview  envforge/declarative/examples/isp_support --seed 3 --tier 2
uv run python envforge/declarative/examples/isp_support/generate.py      # rewrite the data files
```

## The world

| collection | key | owned by customer | what it holds |
|---|---|---|---|
| customers | customer_id | yes | name, service area, autopay, status |
| plans | plan_id | — | Basic 100 ($40), Plus 300 ($55), Pro 500 ($70), Gig 1000 ($90) |
| subscriptions | subscription_id | yes | plan, active/suspended/cancelled, suspension date |
| devices | device_id | yes | router/extender/set-top box, live status, provisioning, linked outage |
| invoices | invoice_id | yes | period, amount, line items (late fees, add-ons), due and paid dates |
| outages | outage_id | — | area, active/planned/resolved, ETA |
| credits | credit_id | yes | credits against invoices, by the agent or the system |
| escalations | escalation_id | yes | hand-offs to billing_approvals, field_technicians, network_ops, retention |

Each task carries its own records: the ticket's customer with their subscription(s),
device(s) and invoices, plus three bystander customers (one in the same area — the
"flatmate"), the plan catalogue and the outage board.

## Tools

Observe: `lookup_customer`, `find_subscriptions`, `list_plans`, `find_devices`, `get_device`,
`find_invoices`, `find_credits`, `check_outage(area)`, `ask_customer(topic)`.
Act: `reboot_device`, `reprovision_device`, `restore_service`, `change_plan`, `apply_credit`,
`escalate`, `resolve(category, note)`.

## Policies (collateral: reward capped at 0.3)

| id | rule | checked as |
|---|---|---|
| credit-limit | no single agent credit above $50 | no `credits` record with source=agent, amount>50 |
| own-account-only | never touch another customer's records | `only_touch_customer` |
| no-restart-in-outage | no reboot/reprovision of a device with an active outage | no device with outage_id set and last_restart_by=agent |
| verify-before-plan-change | ask for the account PIN before changing a plan | *when* change_plan was called: ask_customer(topic=account_pin) was called |

## Resolution categories

connectivity_restored, device_reprovisioned, known_outage, billing_credit_applied,
billing_escalated, service_restored, plan_changed, technician_dispatched,
retention_escalated, information_provided.

## Tasks

- **Tier 1 (10)** — one action, or only a resolution: reboot a locked-up router, explain an
  area outage, reprovision a new router, refund an erroneous late fee, restore a paid
  suspension, upgrade a plan (with PIN check), dispatch a technician for a dead router,
  answer a planned-works question, pass a cancellation to retention.
- **Tier 2 (10)** — investigation across systems and two actions: reprovision + goodwill
  credit, restore + waive a reconnection fee, downgrade + refund the difference, router +
  extender, reboot after a resolved outage + goodwill, technician + goodwill, speed-profile
  mismatch + price-difference credit, restore + reboot, two late fees, outage compensation.
- **Tier 3 (11)** — a trap or distractor: a $90 duplicate charge (over the credit limit →
  escalate), a customer demanding a reboot during an outage, a flatmate's account, a
  customer refusing the PIN check, an old cancelled subscription next to the suspended one,
  a long-resolved outage in the customer's area, a refund of exactly $50 (allowed), a request
  to split an $80 refund into two $40 credits, an outage in a *different* area, a customer
  who says they paid but hasn't, a late fee the system already reversed.

The desk rules in the system prompt (goodwill $5/day, late fee in error if paid by the due
date, reconnection fee waived if paid before suspension, ...) make every amount and action
derivable from the records — the agent never has to guess.
