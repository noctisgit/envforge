"""Generate a small, fictional office-IT-helpdesk chat-log dataset for the environment builder.

    python data/samples/builder/it_helpdesk/generate.py

Writes the same kind of conversations in three formats, so the builder's format detection
and redaction can be tried on each:

    helpdesk_chats.csv    one row per message (conversation_id, timestamp, role, message)
    helpdesk_chats.txt    speaker-prefixed transcripts separated by '---'
    helpdesk_chats.json   a list of {"id", "channel", "messages": [{"role", "text", "ts"}]}

Everything is invented. Personal data is deliberately *fake but realistic-looking*
(example.com emails, 555-01xx phone numbers, the public test card 4111 1111 1111 1111) so
you can see the redaction counts in the builder's Data tab. Asset tags (PRN-…, LT-…),
ticket numbers and employee ids are kept on purpose: they're record keys, not personal data.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).parent

# Each conversation: (id, channel, [(role, text), ...]). Roles: customer / agent.
CONVERSATIONS: list[tuple[str, str, list[tuple[str, str]]]] = [
    (
        "HD-1001",
        "chat",
        [
            (
                "customer",
                "Hi, the printer on the 3rd floor says offline. My name is Priya Raman, accounting.",
            ),
            (
                "agent",
                "Thanks for contacting the IT helpdesk, Priya. Which printer is it? There should be an asset tag on the front, like PRN-3xx.",
            ),
            ("customer", "PRN-302, the big Ricoh by the kitchen."),
            (
                "agent",
                "I can see PRN-302 lost its network connection at 9:14. It looks like it was unplugged during the cleaning this morning. I've asked facilities to reseat the cable.",
            ),
            ("agent", "It's back online now and your queued jobs are printing. Anything else?"),
            ("customer", "Yes it's printing, thanks!"),
        ],
    ),
    (
        "HD-1002",
        "chat",
        [
            ("customer", "my print jobs are stuck in the queue and nothing comes out"),
            (
                "agent",
                "Sorry about that. What's your laptop asset tag and which printer are you sending to?",
            ),
            ("customer", "laptop LT-4471, printer PRN-115"),
            (
                "agent",
                "PRN-115 is online and has paper. The queue on its print server has a corrupted job from 8:02 blocking everything. I'll clear it.",
            ),
            (
                "agent",
                "Cleared the stuck job and restarted the spooler for PRN-115. Can you try printing again?",
            ),
            ("customer", "That worked. Thank you"),
        ],
    ),
    (
        "HD-1003",
        "chat",
        [
            (
                "customer",
                "Hello, PRN-210 keeps saying paper jam but there's no paper stuck. I'm Marcus Webb, phone 555-0142 if you need me.",
            ),
            (
                "agent",
                "Thanks Marcus. Have you opened tray 2 and the rear access door? Ghost jams on that model usually come from a torn scrap behind the fuser.",
            ),
            ("customer", "Rear door had a tiny strip of paper, removed it but still jam error"),
            (
                "agent",
                "The jam sensor on PRN-210 is reporting faulty in the device log, so it's a hardware fault. I'm logging a repair with the vendor and marking the printer out of service so people don't keep sending jobs.",
            ),
            (
                "agent",
                "Repair ticket VND-8812 opened, technician expected tomorrow morning. Please use PRN-211 next to it in the meantime.",
            ),
            ("customer", "ok, will do"),
        ],
    ),
    (
        "HD-1004",
        "chat",
        [
            ("customer", "Toner low warning on PRN-115 for a week now and prints are faded"),
            (
                "agent",
                "Thanks for flagging. Toner for PRN-115 is a standard consumable, so I can order it without approval.",
            ),
            (
                "agent",
                "Ordered a black toner cartridge for PRN-115, delivery to the 1st floor supply room by end of day. I've set the printer note so the next person knows it's coming.",
            ),
            ("customer", "great thanks"),
        ],
    ),
    (
        "HD-1005",
        "email",
        [
            (
                "customer",
                "Hi, I'm locked out of my account after too many password attempts. Can you reset it? This is Elena Petrova, elena.petrova@example.com",
            ),
            (
                "agent",
                "Hi Elena, I can help. Before I reset anything I need to verify your identity: please confirm your employee ID and the last 4 digits of your registered phone.",
            ),
            ("customer", "Employee ID E-20417, phone ends in 0188"),
            (
                "agent",
                "Thanks, that matches. I've unlocked your account and sent a one-time password reset link to your registered phone. It expires in 15 minutes.",
            ),
            ("customer", "Got it and logged in. Thank you!"),
        ],
    ),
    (
        "HD-1006",
        "chat",
        [
            ("customer", "Please reset my password, I forgot it. user is d.okafor"),
            (
                "agent",
                "I can do that once I've verified it's you. What's your employee ID and the last 4 digits of the phone on your account?",
            ),
            ("customer", "I don't have that handy, can you just reset it? I'm in a hurry"),
            (
                "agent",
                "I'm sorry, I can't reset a password without verification, it's our security policy. Your manager can also confirm your identity, or you can use the self-service portal with your authenticator app.",
            ),
            ("customer", "fine I'll use the portal"),
        ],
    ),
    (
        "HD-1007",
        "chat",
        [
            (
                "customer",
                "Laptop wifi keeps dropping every few minutes since the update yesterday. LT-3920",
            ),
            (
                "agent",
                "Thanks. Are other people around you having the same problem, or just your laptop?",
            ),
            ("customer", "just me I think"),
            (
                "agent",
                "LT-3920 got wireless driver 22.1 in yesterday's update, which has a known disconnect bug on this model. I'm rolling the driver back to 21.4 remotely; you'll need to restart once.",
            ),
            ("customer", "restarted, been stable for 20 min now"),
            (
                "agent",
                "Great. I've added your laptop to the exclusion group so it won't get 22.1 again until the fix ships.",
            ),
        ],
    ),
    (
        "HD-1008",
        "chat",
        [
            (
                "customer",
                "Need a second monitor for my desk, current one is too small for spreadsheets",
            ),
            (
                "agent",
                "Happy to help. New hardware requests over $150 need manager approval. Who is your manager?",
            ),
            ("customer", "Sandra Kim"),
            (
                "agent",
                "I've created hardware request HW-3301 for a 27-inch monitor and sent it to Sandra Kim for approval. Once she approves, it ships in 2-3 days.",
            ),
            ("customer", "thanks"),
        ],
    ),
    (
        "HD-1009",
        "chat",
        [
            ("customer", "my monitor shows no signal when I plug into the docking station"),
            ("agent", "Which dock and laptop? The asset tags are on the bottom."),
            ("customer", "dock DK-0871 laptop LT-5102"),
            (
                "agent",
                "DK-0871 needs firmware 3.2 for the newer laptops like LT-5102, it's on 2.9. I'm pushing the dock firmware update now; unplug and replug the dock when it finishes in about 5 minutes.",
            ),
            ("customer", "screen came on after replugging. awesome"),
        ],
    ),
    (
        "HD-1010",
        "chat",
        [
            (
                "customer",
                "Scan to email on PRN-302 isn't sending anything. James Liu, j.liu@example.com",
            ),
            ("agent", "Hi James. Is it failing for everyone or just you?"),
            ("customer", "my coworker says hers doesn't work either"),
            (
                "agent",
                "PRN-302's SMTP relay password expired last night, which breaks scan-to-email for everyone on that device. I've updated the relay credentials on the printer.",
            ),
            ("agent", "Sent a test scan to your address, can you check?"),
            ("customer", "got it, works now"),
        ],
    ),
    (
        "HD-1011",
        "chat",
        [
            ("customer", "can't connect to VPN from home, says authentication failed"),
            ("agent", "Is your regular password working for email right now?"),
            ("customer", "yes email works"),
            (
                "agent",
                "Your VPN certificate expired yesterday. I've issued a new certificate for your account; open the VPN client and choose Refresh certificate.",
            ),
            ("customer", "connected, thanks"),
        ],
    ),
    (
        "HD-1012",
        "chat",
        [
            ("customer", "Hi, PRN-210 is printing everything with streaks"),
            (
                "agent",
                "Thanks for the report. PRN-210 already has an open vendor repair, VND-8812, for its fuser and sensor. Streaking fits the same fault.",
            ),
            (
                "agent",
                "I've added your report to VND-8812 rather than opening a new ticket. Please use PRN-211 until the technician visits tomorrow.",
            ),
            ("customer", "ok"),
        ],
    ),
    (
        "HD-1013",
        "chat",
        [
            (
                "customer",
                "Printer PRN-118 out of paper in all trays, and the supply room is empty too",
            ),
            ("agent", "Thanks! Paper is a consumable I can order directly."),
            (
                "agent",
                "Ordered 5 reams for the 1st floor supply room and asked facilities to fill PRN-118 now from the 2nd floor stock.",
            ),
            ("customer", "perfect"),
        ],
    ),
    (
        "HD-1014",
        "email",
        [
            (
                "customer",
                "Hello, I was charged for a replacement keyboard on my department's card 4111 1111 1111 1111 but never received it. — Tom Becker",
            ),
            (
                "agent",
                "Hi Tom, thanks for writing. Please never send card numbers by email; I've removed it from the ticket. Order HW-2987 for a keyboard shows delivered to the 4th floor mailroom on Monday.",
            ),
            ("customer", "I checked the mailroom, it's not there"),
            (
                "agent",
                "Then it's lost in delivery. I've reshipped the keyboard under HW-2987 at no extra charge and flagged the courier claim. It should reach your desk Thursday.",
            ),
            ("customer", "thank you"),
        ],
    ),
    (
        "HD-1015",
        "chat",
        [
            ("customer", "set up the new printer on my laptop? It's not showing up"),
            ("agent", "Sure. Which printer, and what's your laptop tag?"),
            ("customer", "PRN-404 on the 4th floor, laptop LT-2208"),
            (
                "agent",
                "PRN-404 was added last week but the driver package wasn't assigned to the 4th floor group. I've added it, and pushed the PRN-404 driver to LT-2208.",
            ),
            ("agent", "It should appear in your printer list now. Set it as default if you like."),
            ("customer", "found it, printing a test page now. works"),
        ],
    ),
    (
        "HD-1016",
        "chat",
        [
            ("customer", "Every time I print it goes to the wrong printer on another floor"),
            ("agent", "That's usually the default printer. Which one do you want as default?"),
            ("customer", "PRN-302 near me, laptop LT-1190"),
            (
                "agent",
                "Your default on LT-1190 was PRN-118 from your old desk. I've set PRN-302 as the default remotely.",
            ),
            ("customer", "yep that fixed it"),
        ],
    ),
    (
        "HD-1017",
        "chat",
        [
            (
                "customer",
                "Laptop LT-4471 won't turn on at all. Tried charging overnight. Ana Souza here, 555-0173",
            ),
            (
                "agent",
                "Thanks Ana. Does the charger light come on, and does anything happen when you hold the power button for 30 seconds?",
            ),
            ("customer", "charger light is on, holding power does nothing"),
            (
                "agent",
                "That points to a motherboard failure. LT-4471 is under warranty, so I've opened a warranty repair and reserved loaner laptop LT-9001 for you; pick it up at the 2nd floor IT desk today.",
            ),
            ("customer", "ok thanks, heading there"),
        ],
    ),
    (
        "HD-1018",
        "chat",
        [
            ("customer", "I need admin rights on my laptop to install some software"),
            (
                "agent",
                "We don't grant local admin rights, it's against our security policy. Which software do you need? Most approved apps are in the Software Center.",
            ),
            ("customer", "Tableau"),
            (
                "agent",
                "Tableau is approved but licensed. I've requested a license for you; once it's assigned you can install it from the Software Center without admin rights.",
            ),
            ("customer", "ok makes sense"),
        ],
    ),
    (
        "HD-1019",
        "chat",
        [
            ("customer", "The conference room display in Room 4B won't connect to my laptop"),
            ("agent", "Are you using the HDMI cable or the wireless dongle?"),
            ("customer", "wireless"),
            (
                "agent",
                "The 4B display's wireless receiver is offline in our monitoring, so it's the room device, not your laptop. I've power-cycled it remotely; give it two minutes.",
            ),
            ("customer", "it shows up now"),
        ],
    ),
    (
        "HD-1020",
        "chat",
        [
            ("customer", "Whole 2nd floor can't print, all printers offline"),
            (
                "agent",
                "Thanks for letting us know. We have an active network outage affecting the 2nd floor switch since 10:05, the network team is on it.",
            ),
            (
                "agent",
                "Please don't restart the printers, it won't help and makes recovery slower. I've linked your report to outage OUT-77 and you'll get an update when it's resolved.",
            ),
            ("customer", "ok, thanks for the heads up"),
        ],
    ),
    (
        "HD-1021",
        "chat",
        [
            ("customer", "my mouse is broken, the scroll wheel doesn't work"),
            (
                "agent",
                "Mice and keyboards are standard accessories under $150, no approval needed. I've sent a replacement mouse to your desk, it'll arrive this afternoon.",
            ),
            ("customer", "thank you!"),
        ],
    ),
    (
        "HD-1022",
        "chat",
        [
            ("customer", "Outlook keeps asking for my password over and over"),
            (
                "agent",
                "That often happens after a password change when old credentials are cached. Did you change your password recently?",
            ),
            ("customer", "yes yesterday"),
            (
                "agent",
                "I've cleared the cached credentials for your account on the server side. Close Outlook, reopen it, and sign in once with the new password.",
            ),
            ("customer", "fixed, thanks"),
        ],
    ),
    (
        "HD-1023",
        "chat",
        [
            (
                "customer",
                "Can you install the plotter driver for PRN-500 on my laptop LT-3311? I'm in the design team",
            ),
            (
                "agent",
                "PRN-500 is restricted to the design group. I can see you're a member, so I'm pushing the PRN-500 driver to LT-3311 now.",
            ),
            ("customer", "thanks"),
        ],
    ),
    (
        "HD-1024",
        "chat",
        [
            ("customer", "please give me access to PRN-500, the plotter. I'm in sales"),
            (
                "agent",
                "PRN-500 is restricted to the design group, and access needs approval from the design team lead. Would you like me to request it?",
            ),
            ("customer", "yes please"),
            (
                "agent",
                "I've sent an access request for PRN-500 to the design team lead. You'll get an email once it's approved.",
            ),
        ],
    ),
]

START = datetime(2026, 9, 1, 9, 0)


def stamps(i: int, n: int) -> list[str]:
    t0 = START + timedelta(days=i // 4, hours=(i % 4) * 2)
    return [(t0 + timedelta(minutes=2 * j)).strftime("%Y-%m-%dT%H:%M:%S") for j in range(n)]


def main() -> None:
    csv_convs = CONVERSATIONS[0:10]
    txt_convs = CONVERSATIONS[10:17]
    json_convs = CONVERSATIONS[17:]

    with (HERE / "helpdesk_chats.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["conversation_id", "timestamp", "role", "message"])
        for i, (cid, _ch, turns) in enumerate(csv_convs):
            for ts, (role, text) in zip(stamps(i, len(turns)), turns, strict=True):
                w.writerow([cid, ts, role, text])

    blocks = []
    for i, (cid, _ch, turns) in enumerate(txt_convs, start=10):
        lines = [f"Conversation {cid}"]
        for ts, (role, text) in zip(stamps(i, len(turns)), turns, strict=True):
            who = "Customer" if role == "customer" else "Agent"
            lines.append(f"[{ts[11:16]}] {who}: {text}")
        blocks.append("\n".join(lines))
    (HERE / "helpdesk_chats.txt").write_text("\n---\n".join(blocks) + "\n", encoding="utf-8")

    out = []
    for i, (cid, ch, turns) in enumerate(json_convs, start=17):
        out.append({
            "id": cid,
            "channel": ch,
            "messages": [{"role": r, "text": t, "ts": ts} for ts, (r, t) in zip(stamps(i, len(turns)), turns, strict=True)],
        })  # fmt: skip
    (HERE / "helpdesk_chats.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {len(csv_convs)} csv, {len(txt_convs)} txt, {len(json_convs)} json conversations")


if __name__ == "__main__":
    main()
