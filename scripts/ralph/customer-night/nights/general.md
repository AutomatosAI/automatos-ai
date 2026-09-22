Not a checklist — an evening. Do what a real owner would do, in a sensible order, reacting to what the product gives you. Over the night, make sure all of this happens:

1. **Get acquainted.** Ask Auto what it can do, tell it about the business and what you want handled first, ask what is on the board and what your agents are for. Notice whether it knows things it should (your agents, your tools, your documents).
2. **Build your team.** Create three or four agents of your own — at least one API agent with skills picked from the skills list, at least one new Claude session agent for building things. Try a marketplace agent or team. Edit two of them (rename, change the description, change the model). Delete one that has an open ticket and see what happens to the ticket.
3. **Give real work — at least a dozen tasks by morning.** Mix assignees and ways of asking (Auto vs the board). Examples, in your words, with your details:
   - research with the web: who else sells specialty coffee subscriptions in the UK, and how their pricing compares to yours;
   - a supplier comparison table for compostable coffee bags (three suppliers, price per 250 g bag, lead time);
   - a welcome email for new subscribers; three product descriptions; a blog post about the Ethiopian lot;
   - the Monday dispatch checklist as a one-page SOP;
   - **a one-page website for the roastery** — brief a Claude session agent to build `index.html` + `styles.css` under the deliverables folder, with the subscription plans and a wholesale enquiry form;
   - a spreadsheet (CSV) of the three subscription plans with margin per bag;
   - two documents you write yourself first (a supplier price list CSV, a brand-voice guide) uploaded as knowledge, then questions to Auto and briefs to agents that need them;
   - one mission with two or three agents (e.g. "launch the autumn subscription tier");
   - things a real user does that no test plan lists: a vague one-liner ("sort out the packaging thing"), a brief that contradicts itself, one asking for five things in three lines, a task with yesterday as the deadline, a task for an agent you have just deleted, the same brief to two agents to compare.
   Set `review_mode: "human"` on two of them so they come back to you.
4. **Use the tools.** Ask Auto and the OPS agent for the things the connected tools enable: what is in the calendar next week, draft (do not send) a reply to a supplier email, list recent orders if Shopify is connected. **Never send, post, publish, pay, or delete on any connected app. Drafts only. A calendar entry only if titled `SIM …`, and delete it afterwards.**
5. **Answer what you are asked.** Every 10–15 minutes: `python3 -m tests.sim.customer questions`, answer in character, approve what a business owner would. Note how the question was worded and whether it made sense.
6. **Review what comes back.** Read every deliverable, report and ticket result as it lands — open the files in `{{DELIVERABLES_DIR}}`, open the website's `index.html`. Grade each 1–5 in the diary with one honest line. If it is bad, do what a customer does: send it back with feedback, or ask again. Where a session agent worked, note whether it actually used platform tools (its ticket result and report say so).
7. **Keep going.** Sessions take 10–30 minutes; never sit waiting. Give out the next thing, then check back. Stop *starting* new work at {{STOP_AT}}; use the remaining time to review, answer, and write the report.

