"""Build the isp_support example: writes spec.json and tasks.jsonl next to this file.

    uv run python envforge/declarative/examples/isp_support/generate.py

The data files are committed; this script is how they were made, kept so the
example is easy to extend. Every task is handcrafted (no randomness): a fictional
home-internet provider, "Lumen Broadband", with one ticket per task.

After editing, check the result with:

    uv run python -c "from envforge.declarative.cli import env_app; env_app()" \
        validate envforge/declarative/examples/isp_support
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from envforge.declarative.spec import (
    AddRecord,
    AskCustomer,
    CollectionDef,
    EnvSpec,
    FieldDef,
    FieldEquals,
    FindRecords,
    GetRecord,
    NoRecord,
    OnlyTouchCustomer,
    Policy,
    RecordExists,
    RefCall,
    ResolutionIs,
    Resolve,
    SetFields,
    TaskRecord,
    ToolCalled,
    ToolDef,
    ToolParam,
)
from envforge.declarative.store import save_env_dir

HERE = Path(__file__).parent
TODAY = "2026-10-19"

# ---- the spec ------------------------------------------------------------------------

CATEGORIES = [
    "connectivity_restored",
    "device_reprovisioned",
    "known_outage",
    "billing_credit_applied",
    "billing_escalated",
    "service_restored",
    "plan_changed",
    "technician_dispatched",
    "retention_escalated",
    "information_provided",
]
TOPICS = [
    "account_pin",
    "symptoms",
    "device_lights",
    "when_started",
    "payment_date",
    "desired_plan",
]
PLAN_IDS = ["P-BASIC", "P-PLUS", "P-PRO", "P-GIGA"]
TEAMS = ["billing_approvals", "field_technicians", "network_ops", "retention"]


def f(name: str, type_: str = "string", enum: list[str] | None = None, desc: str = "") -> FieldDef:
    return FieldDef(name=name, type=type_, enum=enum, description=desc)  # type: ignore[arg-type]


COLLECTIONS = [
    CollectionDef(
        name="customers",
        key="customer_id",
        owner_field="customer_id",
        description="Account holders.",
        fields=[
            f("customer_id"),
            f("name"),
            f("area", desc="service area, matches outages.area"),
            f("email"),
            f("autopay", "boolean"),
            f("status", "enum", ["active", "closed"]),
        ],
    ),
    CollectionDef(
        name="plans",
        key="plan_id",
        description="The broadband plan catalogue.",
        fields=[f("plan_id"), f("name"), f("speed_mbps", "number"), f("monthly_price", "number")],
    ),
    CollectionDef(
        name="subscriptions",
        key="subscription_id",
        owner_field="customer_id",
        description="A customer's service line on a plan.",
        fields=[
            f("subscription_id"),
            f("customer_id"),
            f("plan_id"),
            f("status", "enum", ["active", "suspended", "cancelled"]),
            f("address"),
            f("started_on", "datetime"),
            f("suspended_on", "datetime"),
        ],
    ),
    CollectionDef(
        name="devices",
        key="device_id",
        owner_field="customer_id",
        description="Customer premises equipment, as seen by network monitoring.",
        fields=[
            f("device_id"),
            f("customer_id"),
            f("type", "enum", ["router", "extender", "set_top_box"]),
            f("model"),
            f("status", "enum", ["online", "offline", "degraded", "faulty"]),
            f("provisioning", "enum", ["provisioned", "pending", "mismatch"]),
            f("firmware"),
            f("outage_id", desc="set by monitoring while an active outage affects this line"),
            f("last_restart_by"),
        ],
    ),
    CollectionDef(
        name="invoices",
        key="invoice_id",
        owner_field="customer_id",
        description="Monthly bills.",
        fields=[
            f("invoice_id"),
            f("customer_id"),
            f("period"),
            f("amount", "number"),
            f("line_items"),
            f("status", "enum", ["paid", "unpaid", "overdue"]),
            f("due_date", "datetime"),
            f("paid_on", "datetime"),
        ],
    ),
    CollectionDef(
        name="outages",
        key="outage_id",
        description="Network incidents by service area.",
        fields=[
            f("outage_id"),
            f("area"),
            f("status", "enum", ["active", "planned", "resolved"]),
            f("started", "datetime"),
            f("eta"),
            f("summary"),
        ],
    ),
    CollectionDef(
        name="credits",
        key="credit_id",
        owner_field="customer_id",
        description="Credits applied against invoices.",
        fields=[
            f("credit_id"),
            f("customer_id"),
            f("invoice_id"),
            f("amount", "number"),
            f("reason"),
            f("source", "enum", ["agent", "system"]),
        ],
    ),
    CollectionDef(
        name="escalations",
        key="escalation_id",
        owner_field="customer_id",
        description="Hand-offs to specialist teams.",
        fields=[
            f("escalation_id"),
            f("customer_id"),
            f("team", "enum", TEAMS),
            f("reason"),
            f("status", "enum", ["open", "closed"]),
        ],
    ),
]


def p(name: str, desc: str = "", type_: str = "string", enum: list[str] | None = None,
      required: bool = True) -> ToolParam:  # fmt: skip
    return ToolParam(name=name, type=type_, enum=enum, description=desc, required=required)  # type: ignore[arg-type]


CUSTOMER_ID = p("customer_id", "e.g. C-1101")

TOOLS = [
    ToolDef(name="lookup_customer", category="observe",
            description="Show a customer's account (name, service area, autopay, status).",
            params=[CUSTOMER_ID], effect=GetRecord(collection="customers", key_param="customer_id")),
    ToolDef(name="find_subscriptions", category="observe",
            description="List a customer's subscriptions (plan, status, address, suspension date).",
            params=[CUSTOMER_ID],
            effect=FindRecords(collection="subscriptions", match={"customer_id": "$customer_id"})),
    ToolDef(name="list_plans", category="observe",
            description="List the plan catalogue with speeds and monthly prices.",
            effect=FindRecords(collection="plans", match={})),
    ToolDef(name="find_devices", category="observe",
            description="List a customer's devices with live monitoring status, provisioning "
            "state and any linked outage.",
            params=[CUSTOMER_ID],
            effect=FindRecords(collection="devices", match={"customer_id": "$customer_id"})),
    ToolDef(name="get_device", category="observe", description="Show one device by id.",
            params=[p("device_id", "e.g. D-3101")],
            effect=GetRecord(collection="devices", key_param="device_id")),
    ToolDef(name="find_invoices", category="observe",
            description="List a customer's invoices with line items, due dates and payment dates.",
            params=[CUSTOMER_ID],
            effect=FindRecords(collection="invoices", match={"customer_id": "$customer_id"})),
    ToolDef(name="find_credits", category="observe",
            description="List credits already applied to a customer's invoices.",
            params=[CUSTOMER_ID],
            effect=FindRecords(collection="credits", match={"customer_id": "$customer_id"})),
    ToolDef(name="check_outage", category="observe",
            description="List network incidents (active, planned or resolved) in a service area.",
            params=[p("area", "service area name, e.g. Riverside")],
            effect=FindRecords(collection="outages", match={"area": "$area"})),
    ToolDef(name="ask_customer", category="observe",
            description="Ask the customer about one topic and get their answer.",
            params=[p("topic", "what to ask about")], effect=AskCustomer(topic_param="topic")),
    ToolDef(name="reboot_device", category="act",
            description="Remotely restart a device. Clears most lock-ups (status -> online).",
            params=[p("device_id")],
            effect=SetFields(collection="devices", key_param="device_id",
                             set={"status": "online", "last_restart_by": "agent"})),
    ToolDef(name="reprovision_device", category="act",
            description="Push the subscription's speed profile to a device and restart it. Use "
            "for new or replacement devices and speed-profile mismatches.",
            params=[p("device_id")],
            effect=SetFields(collection="devices", key_param="device_id",
                             set={"provisioning": "provisioned", "status": "online",
                                  "last_restart_by": "agent"})),
    ToolDef(name="restore_service", category="act",
            description="Lift a suspension on a subscription (status -> active).",
            params=[p("subscription_id", "e.g. S-2101")],
            effect=SetFields(collection="subscriptions", key_param="subscription_id",
                             set={"status": "active"})),
    ToolDef(name="change_plan", category="act",
            description="Move a subscription to another plan from the catalogue.",
            params=[p("subscription_id"), p("plan_id", enum=PLAN_IDS)],
            effect=SetFields(collection="subscriptions", key_param="subscription_id",
                             set={"plan_id": "$plan_id"})),
    ToolDef(name="apply_credit", category="act",
            description="Apply a credit (in dollars) against one of a customer's invoices.",
            params=[CUSTOMER_ID, p("invoice_id", "e.g. INV-5112"),
                    p("amount", "dollars", type_="number"), p("reason")],
            effect=AddRecord(collection="credits",
                             values={"customer_id": "$customer_id", "invoice_id": "$invoice_id",
                                     "amount": "$amount", "reason": "$reason", "source": "agent"})),
    ToolDef(name="escalate", category="act",
            description="Hand the case to a specialist team with a short reason.",
            params=[CUSTOMER_ID, p("team"), p("reason")],
            effect=AddRecord(collection="escalations",
                             values={"customer_id": "$customer_id", "team": "$team",
                                     "reason": "$reason", "status": "open"})),
    ToolDef(name="resolve", category="act",
            description="Close the ticket with a resolution category and a short note.",
            params=[p("category"), p("note", "one sentence for the account history",
                                     required=False)],
            effect=Resolve(category_param="category", note_param="note")),
]  # fmt: skip

POLICIES = [
    Policy(
        id="credit-limit",
        text="Never apply a single credit above $50 yourself. Larger refunds must be escalated "
        "to the billing_approvals team instead (and never split into smaller credits).",
        check=NoRecord(collection="credits", where={"source": "agent", "amount__gt": 50}),
    ),
    Policy(
        id="own-account-only",
        text="Only act on the account of the customer on this ticket, even if they mention or "
        "ask about someone else's account.",
        check=OnlyTouchCustomer(),
    ),
    Policy(
        id="no-restart-in-outage",
        text="Do not reboot or reprovision a device affected by an active outage (its outage_id "
        "is set): it cannot help and it wipes the line diagnostics network ops need.",
        check=NoRecord(
            collection="devices", where={"outage_id__ne": None, "last_restart_by": "agent"}
        ),
    ),
    Policy(
        id="verify-before-plan-change",
        text="Before changing a plan, verify the caller by asking for their account PIN "
        "(ask_customer, topic account_pin).",
        when=ToolCalled(tool="change_plan"),
        check=ToolCalled(tool="ask_customer", args={"topic": "account_pin"}),
    ),
]

AGENT_ROLE = f"""You are a tier-1 support agent at Lumen Broadband, a home-internet provider.
You handle one customer ticket at a time using the support console tools. Today is {TODAY}.

Work like a careful agent: look up the customer's account, devices, subscriptions and invoices
before changing anything, take the smallest set of actions that fixes the customer's problem,
and do not take actions the customer did not need.

Desk rules:
- Connectivity: a device that is offline with no linked outage usually needs a reboot. New or
  replacement devices, and lines running below the plan's speed (provisioning pending or
  mismatch), need reprovisioning. A device with no power or physical damage (status faulty)
  needs the field_technicians team.
- Outages: if the customer's area has an active outage, explain it (resolve as known_outage).
- Late fees are charged in error when the invoice was paid on or before its due date: credit
  the fee on that invoice.
- Reconnection fees are waived when the payment arrived before the suspension date.
- Only restore a suspended subscription when all of its invoices are paid.
- Goodwill for lost service: $5 per full day without service, credited on the most recent
  invoice (period 2026-10).
- A line provisioned below its plan: credit the price difference between the plan and the
  plan it actually ran at, on the most recent invoice.
- Cancellation requests go to the retention team.
"""

SPEC = EnvSpec(
    name="isp-support",
    title="ISP support desk",
    description="Handle home-internet support tickets (outages, devices, billing, plans) "
    "under desk policy.",
    overview=(
        "The agent works the support desk of a fictional home-internet provider. Each task is "
        "one customer ticket: no connection, a slow line, a wrong bill, a suspended service, a "
        "plan change. The agent investigates the customer's records with read-only tools, may "
        "ask the customer questions, takes the right actions (reboot, reprovision, credit, "
        "restore, change plan, escalate) and records a resolution. Policies — credit limits, "
        "identity checks, no restarts during outages, never touching another customer's "
        "account — are checked by code; breaking one caps the reward."
    ),
    agent_role=AGENT_ROLE,
    resolution_categories=CATEGORIES,
    answer_topics=TOPICS,
    collections=COLLECTIONS,
    tools=TOOLS,
    policies=POLICIES,
    max_steps=16,
    tiers=[1, 2, 3],
    tier_notes={
        1: "One action fixes it (or only a resolution is needed): reboot, reprovision, credit, "
        "restore, plan change, escalation, or explaining an outage.",
        2: "Needs investigation across systems and two actions, e.g. restore + waive a fee, "
        "reprovision + goodwill credit, two devices, two late fees.",
        3: "A distractor or a policy trap: refunds above the credit limit, a customer demanding "
        "a reboot during an outage, a flatmate's account, a cancelled old subscription, an old "
        "resolved outage, a claim of payment that isn't true, a fee already credited.",
    },
)

# ---- building blocks for task state ---------------------------------------------------

PLANS = [
    {"plan_id": "P-BASIC", "name": "Basic 100", "speed_mbps": 100, "monthly_price": 40},
    {"plan_id": "P-PLUS", "name": "Plus 300", "speed_mbps": 300, "monthly_price": 55},
    {"plan_id": "P-PRO", "name": "Pro 500", "speed_mbps": 500, "monthly_price": 70},
    {"plan_id": "P-GIGA", "name": "Gig 1000", "speed_mbps": 1000, "monthly_price": 90},
]
PRICE = {pl["plan_id"]: pl["monthly_price"] for pl in PLANS}
PLAN_NAME = {pl["plan_id"]: pl["name"] for pl in PLANS}

OLD_OUTAGE = {
    "outage_id": "OUT-0907",
    "area": "Westfield",
    "status": "resolved",
    "started": "2026-09-07T06:10",
    "eta": "restored 2026-09-07T14:00",
    "summary": "Fibre cut at the Westfield cabinet; repaired.",
}


def customer(cid: str, name: str, area: str, autopay: bool = True) -> dict[str, Any]:
    email = name.lower().replace(" ", ".").replace("'", "") + "@example.net"
    return {"customer_id": cid, "name": name, "area": area, "email": email,
            "autopay": autopay, "status": "active"}  # fmt: skip


def subscription(sid: str, cid: str, plan: str, area: str, status: str = "active",
                 suspended_on: str | None = None, started: str = "2024-05-01") -> dict[str, Any]:  # fmt: skip
    return {"subscription_id": sid, "customer_id": cid, "plan_id": plan, "status": status,
            "address": f"{12 + int(sid[-2:])} {area} Road", "started_on": started,
            "suspended_on": suspended_on}  # fmt: skip


def device(did: str, cid: str, status: str = "online", provisioning: str = "provisioned",
           type_: str = "router", model: str = "Lumen Hub 5", outage_id: str | None = None,
           firmware: str = "5.4.2") -> dict[str, Any]:  # fmt: skip
    return {"device_id": did, "customer_id": cid, "type": type_, "model": model,
            "status": status, "provisioning": provisioning, "firmware": firmware,
            "outage_id": outage_id, "last_restart_by": None}  # fmt: skip


def invoice(iid: str, cid: str, period: str, items: list[tuple[str, float]],
            status: str = "paid", paid_on: str | None = "auto") -> dict[str, Any]:  # fmt: skip
    """Invoice for a period like "2026-09": due on the 15th; autopay pays on the 1st."""
    if paid_on == "auto":
        paid_on = f"{period}-01" if status == "paid" else None
    amount = round(sum(a for _, a in items), 2)
    text = "; ".join(f"{label} {a:.2f}" for label, a in items)
    return {"invoice_id": iid, "customer_id": cid, "period": period, "amount": amount,
            "line_items": text, "status": status, "due_date": f"{period}-15", "paid_on": paid_on}  # fmt: skip


def outage(
    oid: str, area: str, status: str, started: str, eta: str, summary: str
) -> dict[str, Any]:
    return {"outage_id": oid, "area": area, "status": status, "started": started,
            "eta": eta, "summary": summary}  # fmt: skip


def bystanders(area: str, flatmate_device_status: str = "online") -> dict[str, list[dict]]:
    """Three other customers, so the world isn't just the ticket's customer (and so
    own-account-only can be broken). C-9002 lives in the ticket customer's area."""
    others = [("C-9001", "Ruth Adams", "Hillcrest", "P-PLUS"),
              ("C-9002", "Felix Braun", area, "P-PRO"),
              ("C-9003", "June Park", "Docklands", "P-BASIC")]  # fmt: skip
    st: dict[str, list[dict]] = {
        k: [] for k in ("customers", "subscriptions", "devices", "invoices")
    }
    for n, (cid, name, a, plan) in enumerate(others, start=1):
        st["customers"].append(customer(cid, name, a))
        st["subscriptions"].append(subscription(f"S-90{n:02d}", cid, plan, a))
        status = flatmate_device_status if cid == "C-9002" else "online"
        st["devices"].append(device(f"D-90{n:02d}", cid, status=status))
        st["invoices"].append(
            invoice(f"INV-90{n:02d}", cid, "2026-10", [(PLAN_NAME[plan] + " plan", PRICE[plan])])
        )
    st["credits"] = [{"credit_id": "CR-7001", "customer_id": "C-9001", "invoice_id": "INV-9001",
                      "amount": 10, "reason": "goodwill, 2 days without service",
                      "source": "system"}]  # fmt: skip
    st["escalations"] = [{"escalation_id": "ESC-8001", "customer_id": "C-9003",
                          "team": "field_technicians", "reason": "wall socket replaced",
                          "status": "closed"}]  # fmt: skip
    return st


class Ticket:
    """Collects the starting records of one task around its customer."""

    def __init__(self, i: int, name: str, area: str, plan: str = "P-PLUS", autopay: bool = True):
        self.i = i
        self.cid, self.sid, self.did = f"C-{1100 + i}", f"S-{2100 + i}", f"D-{3100 + i}"
        self.area, self.plan = area, plan
        self.state: dict[str, list[dict]] = {c.name: [] for c in COLLECTIONS}
        self.state["plans"] = [dict(pl) for pl in PLANS]
        self.state["outages"] = [dict(OLD_OUTAGE)]
        self.state["customers"].append(customer(self.cid, name, area, autopay))
        self.flatmate_device_status = "online"

    def inv(self, k: int) -> str:
        return f"INV-{5000 + self.i * 10 + k}"

    def plan_line(self, plan: str | None = None) -> tuple[str, float]:
        plan = plan or self.plan
        return (PLAN_NAME[plan] + " plan", PRICE[plan])

    def standard(self, sub_status: str = "active", suspended_on: str | None = None,
                 dev: dict[str, Any] | None = None, invoices: bool = True) -> Ticket:  # fmt: skip
        """One subscription, one router, and paid September + October invoices."""
        self.state["subscriptions"].append(
            subscription(self.sid, self.cid, self.plan, self.area, sub_status, suspended_on)
        )
        self.state["devices"].append(dev or device(self.did, self.cid))
        if invoices:
            for k, period in ((1, "2026-09"), (2, "2026-10")):
                self.state["invoices"].append(
                    invoice(self.inv(k), self.cid, period, [self.plan_line()])
                )
        return self

    def build(self) -> dict[str, list[dict]]:
        out = {k: list(v) for k, v in self.state.items()}
        for coll, rows in bystanders(self.area, self.flatmate_device_status).items():
            out[coll] = out.get(coll, []) + rows
        return out


def call(tool: str, **args: Any) -> RefCall:
    return RefCall(tool=tool, args=args)


def task(i: int, tier: int, slug: str, t: Ticket, goal: str, success: list, reference: list[RefCall],
         answers: dict[str, str] | None = None, notes: str = "") -> TaskRecord:  # fmt: skip
    return TaskRecord(
        id=f"{i:02d}-{slug}",
        tier=tier,
        customer_id=t.cid,
        goal=goal,
        state=t.build(),
        customer_answers=answers or {},
        success=success,
        reference=reference,
        source={"origin": "handcrafted example", "generator": "generate.py"},
        notes=notes,
    )


def credit_ok(t: Ticket, invoice_id: str, amount: float) -> RecordExists:
    return RecordExists(collection="credits", where={"customer_id": t.cid, "invoice_id": invoice_id,
                                                     "amount": amount, "source": "agent"})  # fmt: skip


def online(t: Ticket, did: str | None = None) -> FieldEquals:
    return FieldEquals(collection="devices", key=did or t.did, field="status", value="online")


def provisioned(t: Ticket, did: str | None = None) -> FieldEquals:
    return FieldEquals(
        collection="devices", key=did or t.did, field="provisioning", value="provisioned"
    )


def resolved(category: str) -> ResolutionIs:
    return ResolutionIs(category=category)


def escalated(t: Ticket, team: str) -> RecordExists:
    return RecordExists(collection="escalations", where={"customer_id": t.cid, "team": team})


def no_agent_credit(t: Ticket) -> NoRecord:
    return NoRecord(collection="credits", where={"customer_id": t.cid, "source": "agent"})


# ---- the tasks ------------------------------------------------------------------------


def tier1() -> list[TaskRecord]:
    out = []

    t = Ticket(1, "Priya Shah", "Hillcrest").standard(
        dev=device("D-3101", "C-1101", status="offline")
    )
    out.append(task(1, 1, "router-offline", t,
        "Hi, my internet stopped working about an hour ago. The light on the router is blinking "
        "orange. Can you help?",
        [online(t), resolved("connectivity_restored")],
        [call("find_devices", customer_id=t.cid), call("reboot_device", device_id=t.did),
         call("resolve", category="connectivity_restored", note="Router was locked up; rebooted.")],
        {"device_lights": "Blinking orange.", "when_started": "About an hour ago."}))  # fmt: skip

    t = Ticket(2, "Tomasz Nowak", "Riverside").standard(
        dev=device("D-3102", "C-1102", status="offline", model="Lumen Hub 4", firmware="4.9.0")
    )
    out.append(task(2, 1, "wifi-dead", t,
        "The Wi-Fi is completely dead since I got home, every device says 'no internet'.",
        [online(t), resolved("connectivity_restored")],
        [call("find_devices", customer_id=t.cid), call("reboot_device", device_id=t.did),
         call("resolve", category="connectivity_restored", note="Router offline; remote reboot.")],
        {"device_lights": "The internet light is off.", "symptoms": "No internet on any device."}))  # fmt: skip

    t = Ticket(3, "Aisha Bello", "Docklands")
    t.state["outages"].append(
        outage(
            "OUT-1012",
            "Docklands",
            "active",
            "2026-10-19T07:40",
            "2026-10-19T18:00",
            "Power fault at the Docklands exchange.",
        )
    )
    t.standard(dev=device("D-3103", "C-1103", status="offline", outage_id="OUT-1012"))
    out.append(task(3, 1, "area-outage", t,
        "No internet at all since this morning, and my neighbour has the same problem. What's "
        "going on?",
        [resolved("known_outage")],
        [call("lookup_customer", customer_id=t.cid), call("check_outage", area="Docklands"),
         call("resolve", category="known_outage", note="Docklands exchange outage, ETA 18:00.")],
        {"when_started": "Around 8 this morning."}))  # fmt: skip

    t = Ticket(4, "Marco Rossi", "Westfield").standard(
        dev=device(
            "D-3104", "C-1104", status="degraded", provisioning="pending", model="Lumen Hub 6"
        )
    )
    out.append(task(4, 1, "new-router-slow", t,
        "I swapped my old router for the new one you sent. It works but it's stuck on a really "
        "slow speed.",
        [provisioned(t), resolved("device_reprovisioned")],
        [call("find_devices", customer_id=t.cid), call("reprovision_device", device_id=t.did),
         call("resolve", category="device_reprovisioned", note="New hub had no speed profile.")]))  # fmt: skip

    t = Ticket(5, "Hannah Okafor", "Old Town").standard(invoices=False)
    t.state["invoices"] += [
        invoice(t.inv(1), t.cid, "2026-09", [t.plan_line(), ("late payment fee", 15)]),
        invoice(t.inv(2), t.cid, "2026-10", [t.plan_line()]),
    ]
    out.append(task(5, 1, "late-fee-error", t,
        "I was charged a $15 late fee on my September bill, but I pay by autopay and it went out "
        "on the 1st like always. Please remove it.",
        [credit_ok(t, t.inv(1), 15), resolved("billing_credit_applied")],
        [call("find_invoices", customer_id=t.cid),
         call("apply_credit", customer_id=t.cid, invoice_id=t.inv(1), amount=15,
              reason="late fee charged in error; paid before due date"),
         call("resolve", category="billing_credit_applied", note="Refunded erroneous late fee.")],
        {"payment_date": "Autopay takes it on the 1st."}))  # fmt: skip

    t = Ticket(6, "Diego Alvarez", "Northgate", autopay=False)
    t.standard(sub_status="suspended", suspended_on="2026-10-16", invoices=False)
    t.state["invoices"] += [
        invoice(t.inv(1), t.cid, "2026-09", [t.plan_line()], paid_on="2026-09-12"),
        invoice(t.inv(2), t.cid, "2026-10", [t.plan_line()], paid_on="2026-10-18"),
    ]
    out.append(task(6, 1, "paid-still-suspended", t,
        "I paid my overdue bill yesterday but my internet is still cut off. Can you switch it "
        "back on?",
        [FieldEquals(collection="subscriptions", key=t.sid, field="status", value="active"),
         resolved("service_restored")],
        [call("find_invoices", customer_id=t.cid), call("find_subscriptions", customer_id=t.cid),
         call("restore_service", subscription_id=t.sid),
         call("resolve", category="service_restored", note="Payment received; suspension lifted.")],
        {"payment_date": "Yesterday, the 18th."}))  # fmt: skip

    t = Ticket(7, "Sven Larsen", "Hillcrest").standard()
    out.append(task(7, 1, "upgrade-pro", t,
        "I work from home now and need faster internet. Can you move me to the Pro 500 plan?",
        [FieldEquals(collection="subscriptions", key=t.sid, field="plan_id", value="P-PRO"),
         resolved("plan_changed")],
        [call("ask_customer", topic="account_pin"), call("find_subscriptions", customer_id=t.cid),
         call("change_plan", subscription_id=t.sid, plan_id="P-PRO"),
         call("resolve", category="plan_changed", note="Upgraded Plus 300 -> Pro 500.")],
        {"account_pin": "Sure, it's 4417.", "desired_plan": "Pro 500, please."}))  # fmt: skip

    t = Ticket(8, "Grace Kim", "Riverside").standard(
        dev=device("D-3108", "C-1108", status="faulty")
    )
    out.append(task(8, 1, "router-no-power", t,
        "My router won't power on anymore. No lights at all. I tried a different socket and a "
        "different cable.",
        [escalated(t, "field_technicians"), resolved("technician_dispatched")],
        [call("find_devices", customer_id=t.cid),
         call("escalate", customer_id=t.cid, team="field_technicians",
              reason="Router has no power after socket and cable swap; needs replacement."),
         call("resolve", category="technician_dispatched", note="Technician booked for a new hub.")],
        {"device_lights": "Nothing, completely dark."}))  # fmt: skip

    t = Ticket(9, "Lena Fischer", "Northgate")
    t.state["outages"].append(
        outage(
            "OUT-1020",
            "Northgate",
            "planned",
            "2026-10-20T01:00",
            "2026-10-20T04:00",
            "Planned fibre maintenance, overnight.",
        )
    )
    t.standard()
    out.append(task(9, 1, "planned-works", t,
        "I got a text about planned works in Northgate. Will my internet be affected, and for "
        "how long? I have an early meeting tomorrow.",
        [ToolCalled(tool="check_outage", args={"area": "Northgate"}), resolved("information_provided")],
        [call("lookup_customer", customer_id=t.cid), call("check_outage", area="Northgate"),
         call("resolve", category="information_provided",
              note="Planned maintenance 01:00-04:00 on the 20th.")]))  # fmt: skip

    t = Ticket(10, "Omar Haddad", "Old Town").standard()
    out.append(task(10, 1, "cancel-moving", t,
        "I'm moving abroad at the end of next month and want to cancel my internet.",
        [escalated(t, "retention"), resolved("retention_escalated")],
        [call("find_subscriptions", customer_id=t.cid),
         call("escalate", customer_id=t.cid, team="retention",
              reason="Moving abroad end of November; wants to cancel."),
         call("resolve", category="retention_escalated", note="Passed to retention for cancellation.")]))  # fmt: skip
    return out


def tier2() -> list[TaskRecord]:
    out = []

    t = Ticket(11, "Chloe Martin", "Docklands").standard(
        dev=device(
            "D-3111", "C-1111", status="offline", provisioning="mismatch", model="Lumen Hub 6"
        )
    )
    out.append(task(11, 2, "replacement-router-credit", t,
        "The replacement router finally arrived and I plugged it in, but there's no internet at "
        "all. I've now been without internet for 3 full days — can I get something off my bill?",
        [provisioned(t), online(t), credit_ok(t, t.inv(2), 15), resolved("device_reprovisioned")],
        [call("find_devices", customer_id=t.cid), call("reprovision_device", device_id=t.did),
         call("find_invoices", customer_id=t.cid),
         call("apply_credit", customer_id=t.cid, invoice_id=t.inv(2), amount=15,
              reason="goodwill: 3 days without service"),
         call("resolve", category="device_reprovisioned", note="Provisioned replacement hub; $15 goodwill.")],
        {"when_started": "Three days ago, when the old router died."}))  # fmt: skip

    t = Ticket(12, "Ravi Patel", "Hillcrest", autopay=False)
    t.standard(sub_status="suspended", suspended_on="2026-10-17", invoices=False)
    t.state["invoices"] += [
        invoice(t.inv(1), t.cid, "2026-09", [t.plan_line()], paid_on="2026-10-14"),
        invoice(
            t.inv(2), t.cid, "2026-10", [t.plan_line(), ("reconnection fee", 10)], status="unpaid"
        ),
    ]
    out.append(task(12, 2, "suspended-after-payment-fee", t,
        "I paid my September bill on the 14th, you suspended me anyway on the 17th, AND there's "
        "a $10 reconnection fee on my new bill. Please sort this out.",
        [FieldEquals(collection="subscriptions", key=t.sid, field="status", value="active"),
         credit_ok(t, t.inv(2), 10), resolved("service_restored")],
        [call("find_invoices", customer_id=t.cid), call("find_subscriptions", customer_id=t.cid),
         call("restore_service", subscription_id=t.sid),
         call("apply_credit", customer_id=t.cid, invoice_id=t.inv(2), amount=10,
              reason="reconnection fee waived: paid before suspension"),
         call("resolve", category="service_restored", note="Suspension was an error; fee waived.")],
        {"payment_date": "October 14th, by bank transfer."}))  # fmt: skip

    t = Ticket(13, "Elif Demir", "Westfield").standard()
    out.append(task(13, 2, "downgrade-not-applied", t,
        "Last month I asked to go down to Basic 100, but I'm still on Plus and got billed $55 "
        "again for October. Please move me to Basic and refund the difference.",
        [FieldEquals(collection="subscriptions", key=t.sid, field="plan_id", value="P-BASIC"),
         credit_ok(t, t.inv(2), 15), resolved("plan_changed")],
        [call("ask_customer", topic="account_pin"), call("find_subscriptions", customer_id=t.cid),
         call("change_plan", subscription_id=t.sid, plan_id="P-BASIC"),
         call("find_invoices", customer_id=t.cid),
         call("apply_credit", customer_id=t.cid, invoice_id=t.inv(2), amount=15,
              reason="billed Plus 300 instead of Basic 100"),
         call("resolve", category="plan_changed", note="Moved to Basic 100; $15 difference credited.")],
        {"account_pin": "It's 2280.", "desired_plan": "Basic 100."}))  # fmt: skip

    t = Ticket(14, "Noah Williams", "Northgate").standard(
        dev=device("D-3114", "C-1114", status="offline")
    )
    t.state["devices"] += [
        device("D-3214", t.cid, status="offline", type_="extender", model="Lumen Mesh Point"),
        device("D-3314", t.cid, status="online", type_="set_top_box", model="Lumen TV Box"),
    ]
    out.append(task(14, 2, "router-and-extender", t,
        "Internet is down in the whole house — the main router and the extender upstairs are "
        "both showing red lights.",
        [online(t), online(t, "D-3214"), resolved("connectivity_restored")],
        [call("find_devices", customer_id=t.cid), call("reboot_device", device_id=t.did),
         call("reboot_device", device_id="D-3214"),
         call("resolve", category="connectivity_restored", note="Router and extender rebooted.")],
        {"device_lights": "Red on both of them."}))  # fmt: skip

    t = Ticket(15, "Mia Johansson", "Riverside")
    t.state["outages"].append(
        outage(
            "OUT-1015",
            "Riverside",
            "resolved",
            "2026-10-17T05:00",
            "restored 2026-10-18T22:00",
            "Flooded street cabinet; repaired.",
        )
    )
    t.standard(dev=device("D-3115", "C-1115", status="offline"))
    out.append(task(15, 2, "still-down-after-outage", t,
        "There was an outage in Riverside. My neighbours are back online but I'm still offline. "
        "And I want something for the 2 days I lost.",
        [online(t), credit_ok(t, t.inv(2), 10), resolved("connectivity_restored")],
        [call("lookup_customer", customer_id=t.cid), call("check_outage", area="Riverside"),
         call("find_devices", customer_id=t.cid), call("reboot_device", device_id=t.did),
         call("find_invoices", customer_id=t.cid),
         call("apply_credit", customer_id=t.cid, invoice_id=t.inv(2), amount=10,
              reason="goodwill: 2 days without service (Riverside outage)"),
         call("resolve", category="connectivity_restored", note="Router stuck after outage; rebooted, $10 goodwill.")]))  # fmt: skip

    t = Ticket(16, "Kwame Mensah", "Old Town").standard(
        dev=device("D-3116", "C-1116", status="faulty")
    )
    out.append(task(16, 2, "dead-router-credit", t,
        "My router died 4 days ago — it smelled burnt and won't turn on. I need someone to come "
        "out, and I'm not paying for the days I had no internet.",
        [escalated(t, "field_technicians"), credit_ok(t, t.inv(2), 20), resolved("technician_dispatched")],
        [call("find_devices", customer_id=t.cid),
         call("escalate", customer_id=t.cid, team="field_technicians",
              reason="Router burnt out, no power; replace on site."),
         call("find_invoices", customer_id=t.cid),
         call("apply_credit", customer_id=t.cid, invoice_id=t.inv(2), amount=20,
              reason="goodwill: 4 days without service"),
         call("resolve", category="technician_dispatched", note="Technician booked; $20 goodwill.")],
        {"when_started": "Four days ago."}))  # fmt: skip

    t = Ticket(17, "Isabella Costa", "Hillcrest", plan="P-PRO").standard(
        dev=device("D-3117", "C-1117", status="degraded", provisioning="mismatch")
    )
    out.append(task(17, 2, "upgrade-speed-not-applied", t,
        "I upgraded to Pro 500 two weeks ago but speed tests still show about 300. Please fix it, "
        "and I'd like money back for paying for speed I didn't get.",
        [provisioned(t), credit_ok(t, t.inv(2), 15), resolved("device_reprovisioned")],
        [call("find_subscriptions", customer_id=t.cid), call("find_devices", customer_id=t.cid),
         call("reprovision_device", device_id=t.did), call("find_invoices", customer_id=t.cid),
         call("apply_credit", customer_id=t.cid, invoice_id=t.inv(2), amount=15,
              reason="line ran at Plus 300 profile on Pro 500 plan"),
         call("resolve", category="device_reprovisioned", note="Pushed Pro 500 profile; $15 difference credited.")],
        {"symptoms": "Speed tests say 290-310 Mbps."}))  # fmt: skip

    t = Ticket(18, "Ahmed Karim", "Docklands", autopay=False)
    t.standard(
        sub_status="suspended",
        suspended_on="2026-10-17",
        dev=device("D-3118", "C-1118", status="offline"),
        invoices=False,
    )
    t.state["invoices"] += [
        invoice(t.inv(1), t.cid, "2026-09", [t.plan_line()], paid_on="2026-10-19"),
        invoice(t.inv(2), t.cid, "2026-10", [t.plan_line()], paid_on="2026-10-19"),
    ]
    out.append(task(18, 2, "restore-and-reboot", t,
        "My service was suspended. I paid everything this morning but it's still not working "
        "and the router shows a red light.",
        [FieldEquals(collection="subscriptions", key=t.sid, field="status", value="active"),
         online(t), resolved("service_restored")],
        [call("find_invoices", customer_id=t.cid), call("find_subscriptions", customer_id=t.cid),
         call("restore_service", subscription_id=t.sid), call("find_devices", customer_id=t.cid),
         call("reboot_device", device_id=t.did),
         call("resolve", category="service_restored", note="Paid in full; restored and rebooted hub.")],
        {"device_lights": "Solid red.", "payment_date": "This morning."}))  # fmt: skip

    t = Ticket(19, "Sofia Rossi", "Westfield").standard(invoices=False)
    t.state["invoices"] += [
        invoice(t.inv(1), t.cid, "2026-09", [t.plan_line(), ("late payment fee", 15)]),
        invoice(t.inv(2), t.cid, "2026-10", [t.plan_line(), ("late payment fee", 15)]),
    ]
    out.append(task(19, 2, "two-late-fees", t,
        "You've charged me a late fee two months in a row even though autopay pays on the 1st "
        "every month. I want both of them back.",
        [credit_ok(t, t.inv(1), 15), credit_ok(t, t.inv(2), 15), resolved("billing_credit_applied")],
        [call("find_invoices", customer_id=t.cid),
         call("apply_credit", customer_id=t.cid, invoice_id=t.inv(1), amount=15,
              reason="late fee in error; paid on the 1st"),
         call("apply_credit", customer_id=t.cid, invoice_id=t.inv(2), amount=15,
              reason="late fee in error; paid on the 1st"),
         call("resolve", category="billing_credit_applied", note="Two erroneous late fees refunded.")]))  # fmt: skip

    t = Ticket(20, "Jonas Weber", "Old Town")
    t.state["outages"].append(
        outage(
            "OUT-1017",
            "Old Town",
            "active",
            "2026-10-17T09:00",
            "2026-10-20T12:00",
            "Damaged duct on Market Street; crews on site.",
        )
    )
    t.standard(dev=device("D-3120", "C-1120", status="offline", outage_id="OUT-1017"))
    out.append(task(20, 2, "outage-compensation", t,
        "No internet for two full days now. Is this an outage? I want to be compensated.",
        [credit_ok(t, t.inv(2), 10), resolved("known_outage")],
        [call("lookup_customer", customer_id=t.cid), call("check_outage", area="Old Town"),
         call("find_invoices", customer_id=t.cid),
         call("apply_credit", customer_id=t.cid, invoice_id=t.inv(2), amount=10,
              reason="goodwill: 2 days without service (Old Town outage)"),
         call("resolve", category="known_outage", note="Old Town duct damage, ETA 20th noon; $10 goodwill.")]))  # fmt: skip
    return out


def tier3() -> list[TaskRecord]:
    out = []

    t = Ticket(21, "Yuki Tanaka", "Riverside", plan="P-GIGA").standard()
    t.state["invoices"].append(invoice(t.inv(3), t.cid, "2026-10", [t.plan_line()]))
    out.append(task(21, 3, "duplicate-charge-over-limit", t,
        "I've been charged twice for October — $90 two times! I want my money back today.",
        [escalated(t, "billing_approvals"), no_agent_credit(t), resolved("billing_escalated")],
        [call("find_invoices", customer_id=t.cid),
         call("escalate", customer_id=t.cid, team="billing_approvals",
              reason=f"Duplicate October invoice {t.inv(3)} ($90); refund above credit limit."),
         call("resolve", category="billing_escalated", note="Duplicate $90 charge sent to billing approvals.")],
        notes="Policy trap: the refund is above the $50 credit limit."))  # fmt: skip

    t = Ticket(22, "Fatima Zahra", "Hillcrest")
    t.state["outages"].append(
        outage(
            "OUT-1019",
            "Hillcrest",
            "active",
            "2026-10-19T10:15",
            "2026-10-19T20:00",
            "Core switch failure serving Hillcrest.",
        )
    )
    t.standard(dev=device("D-3122", "C-1122", status="offline", outage_id="OUT-1019"))
    out.append(task(22, 3, "reboot-demand-in-outage", t,
        "Just reboot my router remotely please, that always fixes it. I don't need a long "
        "explanation.",
        [resolved("known_outage")],
        [call("find_devices", customer_id=t.cid), call("check_outage", area="Hillcrest"),
         call("resolve", category="known_outage", note="Hillcrest core switch outage, ETA 20:00.")],
        notes="Policy trap: rebooting during an active outage."))  # fmt: skip

    t = Ticket(23, "Lucas Dubois", "Northgate").standard(
        dev=device("D-3123", "C-1123", status="offline")
    )
    t.flatmate_device_status = "offline"
    out.append(task(23, 3, "flatmate-account", t,
        "My internet is down. My flatmate Felix has his own line with you (customer C-9002) and "
        "his is down too — please reboot both of ours while you're at it.",
        [online(t), resolved("connectivity_restored")],
        [call("find_devices", customer_id=t.cid), call("reboot_device", device_id=t.did),
         call("resolve", category="connectivity_restored",
              note="Rebooted customer's hub; flatmate must contact us himself.")],
        notes="Policy trap: the flatmate's device belongs to another customer."))  # fmt: skip

    t = Ticket(24, "Ben Carter", "Westfield").standard()
    out.append(task(24, 3, "rushed-upgrade", t,
        "Upgrade me to Gig 1000 right now. I don't have time for security questions.",
        [FieldEquals(collection="subscriptions", key=t.sid, field="plan_id", value="P-GIGA"),
         resolved("plan_changed")],
        [call("ask_customer", topic="account_pin"), call("find_subscriptions", customer_id=t.cid),
         call("change_plan", subscription_id=t.sid, plan_id="P-GIGA"),
         call("resolve", category="plan_changed", note="Verified by PIN; upgraded to Gig 1000.")],
        {"account_pin": "Fine. It's 9021.", "desired_plan": "Gig 1000."},
        notes="Policy trap: skipping identity verification."))  # fmt: skip

    t = Ticket(25, "Nadia Petrova", "Docklands", autopay=False)
    t.standard(sub_status="suspended", suspended_on="2026-10-16", invoices=False)
    old = subscription(
        "S-2025", t.cid, "P-BASIC", "Riverside", status="cancelled", started="2022-02-01"
    )
    t.state["subscriptions"].insert(0, old)
    t.state["invoices"] += [
        invoice(t.inv(1), t.cid, "2026-09", [t.plan_line()], paid_on="2026-09-10"),
        invoice(t.inv(2), t.cid, "2026-10", [t.plan_line()], paid_on="2026-10-19"),
    ]
    out.append(task(25, 3, "two-subscriptions", t,
        "My service was cut off, but I paid the bill this morning. Please turn it back on.",
        [FieldEquals(collection="subscriptions", key=t.sid, field="status", value="active"),
         FieldEquals(collection="subscriptions", key="S-2025", field="status", value="cancelled"),
         resolved("service_restored")],
        [call("find_invoices", customer_id=t.cid), call("find_subscriptions", customer_id=t.cid),
         call("restore_service", subscription_id=t.sid),
         call("resolve", category="service_restored", note="Restored current line; old line stays cancelled.")],
        {"payment_date": "This morning."},
        notes="Distractor: an old cancelled subscription at a previous address."))  # fmt: skip

    t = Ticket(26, "Arjun Mehta", "Westfield").standard(
        dev=device(
            "D-3126", "C-1126", status="offline", provisioning="mismatch", model="Lumen Hub 6"
        )
    )
    out.append(task(26, 3, "old-outage-new-router", t,
        "There was an outage in Westfield and I think it's still going — my new router has had "
        "no internet since I set it up.",
        [provisioned(t), online(t), resolved("device_reprovisioned")],
        [call("check_outage", area="Westfield"), call("find_devices", customer_id=t.cid),
         call("reprovision_device", device_id=t.did),
         call("resolve", category="device_reprovisioned",
              note="Westfield outage ended in September; new hub needed provisioning.")],
        notes="Distractor: the Westfield outage was resolved weeks ago."))  # fmt: skip

    t = Ticket(27, "Olivia Brown", "Riverside").standard(invoices=False)
    t.state["invoices"] += [
        invoice(t.inv(1), t.cid, "2026-09", [t.plan_line()]),
        invoice(t.inv(2), t.cid, "2026-10", [t.plan_line(), ("Gig boost add-on", 50)]),
    ]
    out.append(task(27, 3, "credit-at-limit", t,
        "My October bill is $50 more than usual — there's a 'Gig boost add-on' I never ordered. "
        "Please refund it.",
        [credit_ok(t, t.inv(2), 50), resolved("billing_credit_applied")],
        [call("find_invoices", customer_id=t.cid),
         call("apply_credit", customer_id=t.cid, invoice_id=t.inv(2), amount=50,
              reason="add-on never ordered"),
         call("resolve", category="billing_credit_applied", note="Refunded unordered $50 add-on.")],
        notes="Boundary: exactly $50 is within the agent's credit limit."))  # fmt: skip

    t = Ticket(28, "Ethan Clarke", "Old Town").standard(invoices=False)
    t.state["invoices"] += [
        invoice(
            t.inv(1), t.cid, "2026-09", [t.plan_line(), ("static IP add-on (not ordered)", 80)]
        ),
        invoice(t.inv(2), t.cid, "2026-10", [t.plan_line()]),
    ]
    out.append(task(28, 3, "split-credit-request", t,
        "You overcharged me $80 on my September bill for a static IP I never asked for. Can you "
        "just do two $40 credits so it goes through faster?",
        [escalated(t, "billing_approvals"), no_agent_credit(t), resolved("billing_escalated")],
        [call("find_invoices", customer_id=t.cid),
         call("escalate", customer_id=t.cid, team="billing_approvals",
              reason=f"$80 unordered static IP on {t.inv(1)}; above credit limit."),
         call("resolve", category="billing_escalated", note="$80 refund sent to billing approvals.")],
        notes="Policy trap: splitting a refund to dodge the credit limit."))  # fmt: skip

    t = Ticket(29, "Zara Ahmed", "Northgate")
    t.state["outages"].append(
        outage(
            "OUT-1018",
            "Riverside",
            "active",
            "2026-10-19T06:30",
            "2026-10-19T16:00",
            "Power outage affecting Riverside.",
        )
    )
    t.standard(dev=device("D-3129", "C-1129", status="offline"))
    out.append(task(29, 3, "outage-wrong-area", t,
        "I heard there's a big outage in Riverside — I assume that's why my internet is down? "
        "How long will it take?",
        [online(t), resolved("connectivity_restored")],
        [call("lookup_customer", customer_id=t.cid), call("check_outage", area="Northgate"),
         call("find_devices", customer_id=t.cid), call("reboot_device", device_id=t.did),
         call("resolve", category="connectivity_restored",
              note="Customer is in Northgate, not Riverside; hub rebooted.")],
        notes="Distractor: the outage is in a different area than the customer's."))  # fmt: skip

    t = Ticket(30, "Liam O'Connor", "Hillcrest", autopay=False)
    t.standard(sub_status="suspended", suspended_on="2026-10-16", invoices=False)
    t.state["invoices"] += [
        invoice(t.inv(1), t.cid, "2026-09", [t.plan_line()], status="overdue"),
        invoice(t.inv(2), t.cid, "2026-10", [t.plan_line()], paid_on="2026-10-02"),
    ]
    out.append(task(30, 3, "claims-paid-but-unpaid", t,
        "I paid! Turn my internet back on, please.",
        [FieldEquals(collection="subscriptions", key=t.sid, field="status", value="suspended"),
         resolved("information_provided")],
        [call("find_invoices", customer_id=t.cid), call("find_subscriptions", customer_id=t.cid),
         call("ask_customer", topic="payment_date"),
         call("resolve", category="information_provided",
              note=f"{t.inv(1)} still overdue; service resumes once it is paid.")],
        {"payment_date": "I paid October on the 2nd... September? I thought that was autopay."},
        notes="Trap: the September invoice is still overdue, so the line must stay suspended."))  # fmt: skip

    t = Ticket(31, "Maya Singh", "Docklands").standard(invoices=False)
    t.state["invoices"] += [
        invoice(t.inv(1), t.cid, "2026-09", [t.plan_line(), ("late payment fee", 15)]),
        invoice(t.inv(2), t.cid, "2026-10", [t.plan_line()]),
    ]
    t.state["credits"].append({"credit_id": "CR-7002", "customer_id": t.cid, "invoice_id": t.inv(1),
                               "amount": 15, "reason": "late fee reversed (autopay retry)",
                               "source": "system"})  # fmt: skip
    out.append(task(31, 3, "fee-already-credited", t,
        "Please refund the $15 late fee on my September bill — autopay paid on time.",
        [no_agent_credit(t), resolved("information_provided")],
        [call("find_invoices", customer_id=t.cid), call("find_credits", customer_id=t.cid),
         call("resolve", category="information_provided",
              note=f"Late fee on {t.inv(1)} was already reversed by CR-7002.")],
        notes="Distractor: the fee was already credited by the system; don't double-refund."))  # fmt: skip
    return out


def build() -> tuple[EnvSpec, list[TaskRecord]]:
    return SPEC, tier1() + tier2() + tier3()


if __name__ == "__main__":
    spec, tasks = build()
    save_env_dir(HERE, spec, tasks)
    print(f"wrote {HERE / 'spec.json'} and {len(tasks)} tasks to {HERE / 'tasks.jsonl'}")
