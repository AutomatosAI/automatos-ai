This session is about one thing: **the same work, different hands.** Your agents can run in different ways — as Claude sessions, as Codex sessions, or as API agents on other models (OpenAI, DeepSeek, Kimi and whatever else the model list offers). Tonight you give identical briefs to each kind and keep score: who did it well, who did it fast, and what each cost. You are not choosing a favourite — you are building the comparison table a real owner would want before deciding who gets what.

Do it in this order, reacting to what comes back.

1. **Build the panel.** Look at the model list and the agent editor. Create one agent per runtime you can find — at least: one Claude session agent, one Codex session agent, and two or three API agents on different models (pick the cheap one, the expensive one, and one you have never heard of). Give them the same job title and the same short description. Record for each: how it was created, what you had to choose, and anything you had to guess (a model you were not sure was working, a setting with no explanation).

2. **Keep a runtime ledger** — the most important thing you write. Add a section `## Runtime ledger` to the morning report, one row per brief per agent:
   `| # | brief | agent (runtime / model) | started → finished (minutes) | did it finish? | quality 1–5 (one line why) | did it use the platform (documents, tools) or only its own head? | asks raised | cost if the product shows it | your verdict: keep / not for this |`

3. **Five identical briefs to everyone.** Send the same five pieces of work to every agent on the panel, word for word: a supplier comparison table (three suppliers, price per 250 g bag, lead time — with your price list uploaded), three product descriptions in your brand voice, the Monday dispatch checklist as a one-page SOP, a reply — draft only — to a café asking for 60-day terms, and a question only your documents can answer. Then read every result side by side, brief by brief.

4. **One mission across runtimes.** Set up a single mission staffed by agents of different runtimes (e.g. "launch the autumn tier": the API agent researches, the Claude session writes, the Codex session builds the web page), where the later steps need what the earlier ones found. Did the facts cross from one runtime's work to the next? Record each fact: carried / lost / changed.

5. **Break it once.** Give one brief to an API agent on a model you suspect does not work. What are you told — a clear "this model is unavailable", a silent failure, or a ticket that sits forever? Quote it.

6. **Keep going.** Answer what you are asked (`python3 -m tests.sim.customer questions`) every 10–15 minutes. **Never send, post, publish, pay or delete on any connected app — drafts only.** Stop *starting* new work at {{STOP_AT}}; use the remaining time to finish the ledger, open every file, and write the report — its first section is the comparison table, and the second is who you would give each kind of work to and why.

Outputs land under `{{DELIVERABLES_DIR}}`; everything you write is fictional, at `.example` addresses.
