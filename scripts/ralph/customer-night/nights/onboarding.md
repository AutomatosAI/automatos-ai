This session is about one thing: **the first day.** You have just signed up. Nothing is set up: no agents, no documents, no connected apps beyond what the operator pre-connected. You are going to be onboarded by Auto, accept what it proposes, and get through a real first day of work — and find out how long it takes before this product does something useful for you, and how much you had to work out on your own.

Do it in this order, reacting to what comes back.

1. **Arrive.** Say hello to Auto the way a new customer does — who you are, what the business is, what you want handled — in two or three messages, not a briefing document. Let it lead. Answer its questions honestly and briefly; when it asks something a real owner would not know how to answer (jargon, a setting), say so and note it.

2. **Keep an onboarding ledger** — the most important thing you write. Add a section `## Onboarding ledger` to the morning report and log, in order, every step with its clock time:
   `| # | time | what Auto asked or proposed | what you answered | what happened next (what got created, what you had to click) | how long you waited | did you understand it? (yes / partly / no — one line) |`
   And keep these three numbers at the top: **time to first useful thing** (a document, an answer, a plan you would actually use), **time to first agent doing work**, **things you had to figure out alone**.

3. **Accept the package.** When Auto proposes a package (a set of agents, playbooks, tools for a business like yours), read what it says it will install before accepting. Accept it. Then check the board, the agents page and the playbooks: does what got installed match what was described? Anything installed you did not ask for? Anything promised and missing? Every difference is a row.

4. **Give it a real morning.** With whatever the package set up, do a normal Monday: ask for the dispatch checklist, hand out two pieces of work, upload the two documents you always have to hand (the green-coffee price list as CSV, the brand-voice guide as Markdown — write them first under `{{DELIVERABLES_DIR}}/onboarding-{{DATE}}/`) and ask a question only they can answer. Note whether Auto now uses your documents or still behaves as if you were mid-setup.

5. **Try to finish.** Ask Auto whether setup is done, and what is left. Is there a clear end to onboarding, or does it keep pulling you back? Can you tell where you are in it from the screen?

6. **Answer what you are asked.** Every 10–15 minutes: `python3 -m tests.sim.customer questions`. A question that arrives before you know what an "agent" is tells you something — note how it was worded.

7. **Keep going.** Stop *starting* new work at {{STOP_AT}}; use the remaining time to finish the ledger, open every output, and write the report — the report's first section this time is the three numbers and what you would tell a friend who is about to sign up.

**Never send, post, publish, pay or delete on any connected app — drafts only.** Everything you write is fictional, at `.example` addresses.
