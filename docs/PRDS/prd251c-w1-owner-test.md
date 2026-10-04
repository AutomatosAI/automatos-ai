# PRD-251C Wave 1: the owner's test (research from the first plan, and history)

Wave 1 was written directly on `feat/prd-251c-w1-research-history` (no Ralph loop). It is stacked on `feat/socials-flex`. Run this on the local stack once the branch is merged into local main, the backend is restarted and the frontend is rebuilt, and before Wave 2 starts. Record what failed in the PR.

The defaults this wave took while O5 is open:

- the repeat window is 60 days;
- a person's close topic only warns;
- two titles are the same topic when their content words overlap at 0.75 or more, so "Web Summit in 4 weeks" after "in 5 weeks" is not a repeat.

## 0. Set up

1. **Local main** carries `feat/socials-flex` and this branch. The backend runs from the bind mount, so restart it. TESTER rebuilds the frontend; never start a build yourself.
2. **The boot seed:** after the restart, the marketplace's **Content bank research** Playbook step mentions the history (Marketplace → Playbooks → Content bank research → its step). It held PRD-251B's prompt before. A prompt someone edited there would have been left alone.
3. **Workspace c1:** Socials on, no Socials package installed, so Playbooks lists no "Content bank research". c1 already has plans from before this PRD.

## 1. Research comes with a plan (US-C101)

- [ ] **c1, an existing plan:** open it and press **Save plan** without changes. Playbooks now lists **Content bank research**, and its step runs the Social Media Director (installed with it if it was missing).
- [ ] **Save again:** nothing new is installed; there is still one "Content bank research".
- [ ] **The Content bank step** shows no "Research is not set up" note.
- [ ] **An edited copy stays:** change the playbook's prompt in Playbooks, save the plan again, and your edit is still there.
- [ ] **A deleted copy stays deleted:** delete the playbook in Playbooks and save the plan. It is not put back, and the Content bank says "Research is off: its playbook was removed. Research again restores it."

## 2. Research again restores it (US-C102)

- [ ] **With the playbook deleted, press Research again:** the playbook is back in Playbooks and a research run starts (the toast says research started).
- [ ] **Plan with Auto in a workspace that never had Socials:** say what the week should do and save. The plan saves, Auto's ideas join the bank, and research starts with no "Research did not start" toast.
- [ ] **The weekly run never installs:** on a plan whose playbook you deleted, set research to today, a few minutes ahead. At that time you get one "Research could not run: … Research again restores it." notification, and the playbook is still gone.

## 3. History (US-C103)

- [ ] **`/api/socials/history` in the browser** (signed in) lists the workspace's posts that went out, are approved or scheduled, or wait for approval, newest first. Each shows its title, topic, channels, opening line, date and state. Drafts are not listed.
- [ ] **Delete a plan that made posts:** its posts are still in the history, now with no plan, and each one's topic is the first line of its brief.

## 4. No repeats (US-C104)

- [ ] **A person's repeat warns:** a post titled "What is a Mission?" went out in the last 60 days. Add "what's a mission" to a plan's bank by hand. It is added, with the toast `Added. Posted <date> as "What is a Mission?".`
- [ ] **Research's repeats are refused:** after a research run, open its output in Playbooks. Any idea close to a post or a banked topic is listed as refused, "too close to …", with the earlier one's date.
- [ ] **The repeat window:** What to research → **Not again for (days)**. Set it to 30, save and reload: it reads 30.
- [ ] **A save keeps the last research run** (the fix in this wave): save the plan, and research does not start again at the next tick.

## 5. Research and the composer read the history (US-C105)

- [ ] **Research's first read** (the run's transcript): `platform_get_social_plan` answers with `history`.
- [ ] **The composer:** New post → **Redraft with Auto** on a brief like a recent post's. The opening line differs from the recent posts' openings.
- [ ] **Research's Deliverables:** the run's `platform_list_deliverables` call passes `exclude_source_types ["social_post"]`, so no Socials post render comes back as new material. Asking Auto to list the workspace's deliverables still lists them, as before.

## 6. The bank shows near-repeats (US-C106)

- [ ] **A close topic says so:** a topic close to a post shows `Posted <date> as "…".` under its title, as a link. The link opens that post.
- [ ] **Its own post never counts:** once a post is made from a topic, that topic does not point at its own post.
