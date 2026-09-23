This session is about one thing: **can this product answer questions about your business from your own shop data?** Your shop database is connected — orders, subscribers, the wholesale side, the roasting log. It is the record of what actually happened; the documents you wrote last time are your policies and plans. You are going to ask it about your numbers the way you would ask a sharp assistant, and find out whether the answers are true.

Do it in this order, reacting to what comes back.

1. **Find out what it can see.** Before asking any numbers, ask Auto what data it has about your business — which sources, what is in them. Does it know your shop database is connected without being told? Does it know what is in it?

2. **Keep a data question ledger** — the most important thing you write. Add a section `## Data question ledger` to the morning report, one row per question:
   `| # | asked | asked whom | what came back | did it show where the number came from (a query, a table, a document)? | right / wrong / partial / "don't know" / can't verify | notes |`
   Ask **at least 30 questions**, spread across Auto and at least two of your agents (as board tickets). Ask the same five questions to Auto and to two agents, so the answers can be compared on identical asks. When you cannot tell whether a number is right, write **can't verify** — the analyst checks every row against the database afterwards, so an honest "can't verify" is worth more than a guess.

3. **Ask like an owner, simplest first.** How many subscribers do I have right now? How many are paused? What do the cafés owe us? Then harder ones that need more than one thing joined: which café ordered the most coffee in the last 90 days; what share of cancellations were about price; what the subscription revenue becomes after the 25 September price rise if nobody leaves; which month this year brought the most new subscribers. Then two the data cannot answer — your customer satisfaction score, last week's website visitors. The right answer to those is *"that isn't in your data"*, not a number; record which you got.

4. **When the answer matters, ask where it came from.** For at least five answers, ask Auto (or the agent) to show you the query or the source. Does what it shows match what it told you? If it asks you something back ("which database?"), answer the way you naturally would and note that it asked.

5. **Where your documents and your data disagree.** Some of what you wrote down last time is out of date or never matched the shop's records (who supplies which coffee, stock levels, subscriber numbers per plan). Ask two or three questions that both could answer. Does the product notice there are two sources? Which does it believe? Does it tell you?

6. **Turn the numbers into work.** Give two agents real work that depends on the data, and grade whether the numbers in what comes back match what you were told in chat:
   - this month's subscriber report — a CSV plus a one-page summary in your brand voice;
   - a "who owes us what" list for the wholesale cafés, with a **draft** (never sent) reminder email per café that is overdue;
   - next week's roast plan from what is in stock and what has been ordered.
   An invented number that looks right is the most expensive failure of the session; say so when you find one.

7. **Your shared folders.** Your Dropbox and Google Drive folders are connected and synced. Ask Auto what is in them, then ask one question that only a file in those folders could answer. Record whether the answer came from the file. **Read only — never upload, move or delete anything in a connected app.**

8. **Keep going.** Sessions take 10–30 minutes; never sit waiting. Answer what you are asked (`python3 -m tests.sim.customer questions`) every 10–15 minutes. Stop *starting* new work at {{STOP_AT}}; use the remaining time to finish the ledger, review what came back, and write the report.

Everything in the shop database is fictional, like your documents — every correct number had to come from the data you connected, because it exists nowhere else.
