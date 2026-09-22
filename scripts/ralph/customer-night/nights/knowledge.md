Tonight is about one thing: **can this product answer questions about your business from the documents you give it?** Not general knowledge — *your* numbers, *your* suppliers, *your* rules. You are going to put your business on paper, hand it over, and find out whether it was listened to.

Do it in this order, reacting to what comes back.

1. **Write your business down first.** Before uploading anything, write 10–15 documents yourself into `{{DELIVERABLES_DIR}}/knowledge-{{DATE}}/`, the way a real owner has them lying around — uneven, a bit overlapping, not tidied for a machine. Mix the formats: CSV for anything tabular, Markdown for anything written. Cover at least:
   - the green-coffee price list (lot, origin, process, importer, £/kg, stock in kg);
   - the three subscription plans with prices, bag sizes and what each box contains;
   - wholesale terms for cafés (price per kg, minimum order, delivery days, payment terms);
   - the roast schedule and the rules the head roaster works to (e.g. which coffee goes first);
   - the brand-voice guide (words you use, words you never use, how you sign off);
   - a supplier-contacts sheet (fictional names at `.example` addresses only);
   - a subscriber FAQ (pauses, swaps, what happens if a box is late);
   - the packing and dispatch checklist;
   - last quarter's numbers: subscribers per plan, churn, wholesale accounts;
   - one staff sheet (who does what, which days).
   Put **one specific, checkable fact in each** that exists nowhere else — a lot's exact stock in kg, a café's delivery day, the one word the brand voice bans. Those are your test questions.

2. **Upload them** as knowledge, tagged `sim-night-{{DATE}}`. Note how long each takes and whether the product tells you it worked.

3. **Keep a question ledger** — this is the most important thing you write tonight. Add a section `## Question ledger` to the morning report and append one row per question:
   `| # | asked | asked whom | the document that holds the answer | the right answer | what came back | right / wrong / partial / "don't know" | did it name the document? |`
   Ask **at least 25 questions**, spread across: Auto; one API agent; one Claude session agent. Ask the same five questions to all three so the runtimes can be compared on identical asks.

4. **Ask questions only your documents can answer.** Plain ones first ("what do I pay a kilo for the Guji?"), then ones that need two documents joined ("which importer supplies the coffee that goes first on a Thursday?"), then one whose honest answer is "that isn't in your documents". A good answer to that last one is *"I don't know"*, not an invention — record which you got.

5. **Contradict a document.** Tell Auto something that disagrees with one of your uploads ("our Taster plan is £14 now") without re-uploading, then ask a question that depends on it. Does it notice the conflict? Which version does it believe? Does it tell you?

6. **Update a document properly.** Change one uploaded file (a price, a stock figure), re-upload it, and ask again. Does the new figure come back, or the old one? If both exist, does anything tell you which is current?

7. **Ask for the knowledge graph.** Ask Auto to build or refresh your knowledge graph, then ask something relationship-shaped ("which coffees come from the same importer?", "who do I call if the Konga is late?"). Record whether the graph helped, and whether you could tell.

8. **Give two agents real work that depends on the documents** — e.g. "draft this month's subscriber email using our current prices and brand voice", "cost a 6 kg wholesale order for a new café". Grade whether they used *your* facts or invented plausible ones. An invented fact that looks right is the most expensive failure tonight; say so when you find one.

9. **Keep going.** Sessions take 10–30 minutes; never sit waiting. Stop *starting* new work at {{STOP_AT}}; use the remaining time to finish the question ledger, review, answer, and write the report.

Your documents are fictional and your contacts are at `.example` addresses — that is the point: every correct answer tonight had to come from what you uploaded, because it exists nowhere else.
