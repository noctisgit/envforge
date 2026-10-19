# Sample: office IT helpdesk chat logs (fictional)

24 invented helpdesk conversations about printers, laptops, docks, accounts and access —
for trying the environment builder. The same kind of data in three formats:

| file | conversations | layout |
|---|---|---|
| `helpdesk_chats.csv` | 10 | one row per message: `conversation_id,timestamp,role,message` |
| `helpdesk_chats.txt` | 7 | `Conversation HD-…` headers, `[hh:mm] Customer:` / `Agent:` lines, `---` separators |
| `helpdesk_chats.json` | 7 | `[{"id", "channel", "messages": [{"role", "text", "ts"}]}]` |

Built-in themes the builder can turn into records, tools and policies: printers (offline,
jams, toner/paper, drivers, scan-to-email, default printer), laptops (wifi driver rollback,
warranty + loaner), docks and displays, VPN certificates, Outlook credentials, outages
(don't restart devices during OUT-77), hardware requests (manager approval over $150),
password resets (identity verification required), restricted printers and admin rights.

Personal data is fake but realistic (example.com emails, 555-01xx phones, test card
4111 1111 1111 1111) so the Data tab shows redaction at work. Regenerate with
`python data/samples/builder/it_helpdesk/generate.py`.
