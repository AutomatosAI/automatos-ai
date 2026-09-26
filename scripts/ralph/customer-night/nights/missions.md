This session is about one thing: **can you give this product a job that takes several people, and trust what comes back?** A Mission is a piece of work Automatos plans and staffs across your agents — it proposes a plan, you approve it, the agents do their parts, and it pulls the results together. You are going to run three of increasing size, tell it exactly who does what, and see whether the second job remembers what the first one learned.

Do it in this order, reacting to what comes back.

1. **A small one first.** Ask Auto for a mission with a clear, bounded outcome — e.g. *"prepare the October wholesale price letter: check current café prices, draft the letter in our voice, and list who gets it"*. Read the plan it proposes before approving: right steps, right agents, anything missing? Approve it. Note how long from approval to the first agent starting, and whether you could tell what was happening while it ran. Launch every mission this way — by asking Auto in chat, never from the Missions page: a mission Auto launches carries a watch that scores the result against your request. When the watch reports its verdict (or when you ask Auto for it), note the score and whether you agree.

2. **Keep a mission ledger** — the most important thing you write. Add a section `## Mission ledger` to the morning report, one row per mission:
   `| # | mission | plan quality (1–5, one line why) | did staffing follow your words? | approval flow (what you were asked, when) | time to first agent / to done | deliverables (open each; 1–5) | synthesis: did the final answer use every part? | watch verdict (score vs threshold) — do you agree? | what the second mission remembered from the first |`

3. **A medium one, with your words for the staffing.** Say who does what, explicitly: *"the WRITER drafts, COUNTINGHOUSE checks every number against the shop data, OPS books nothing but lists the calendar slots"*. Does the plan obey you, or reassign? If it reassigns, say so in the ledger word for word.

4. **The one that must remember.** Design the third mission so it can only be done well with facts the first two produced (the café prices you confirmed, the slots OPS listed, the numbers COUNTINGHOUSE checked). Do not repeat those facts in the brief. Then read every deliverable for them: were they recalled by a *different* agent than the one that learned them, were they right, and did anything get "remembered" that was never true? Record each fact: recalled correctly / not recalled / recalled wrong.

5. **Break the flow once.** Reject one plan with feedback ("swap the two agents", "drop step 3") and see whether the next plan reflects it. Let one mission run without approving it for 20 minutes — does it wait, nag, or start anyway?

6. **Read the synthesis like a customer.** For each mission, compare the final write-up with the individual deliverables: is anything in the summary that no agent produced? Is anything an agent produced missing from it? An invented line in a synthesis is the most expensive failure of the session; quote it if you find one. Then set the watch's verdict against your own read: a watch that passes a mission you would reject, or fails one you would accept, is a finding — quote both.

7. **Keep going.** Missions take 20–60 minutes; never sit waiting. Answer what you are asked (`python3 -m tests.sim.customer questions`) every 10–15 minutes — a mission that needs you and does not get you tells you something too. **Never send, post, publish, pay or delete on any connected app — drafts only.** Stop *starting* new work at {{STOP_AT}}; use the remaining time to finish the ledger, open every deliverable, and write the report.

Outputs land under `{{DELIVERABLES_DIR}}`; every fact is fictional and at `.example` addresses — a correct recall had to come from what an earlier mission actually produced, because it exists nowhere else.
