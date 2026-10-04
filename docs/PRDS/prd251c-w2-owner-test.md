# PRD-251C Wave 2: the owner's test (weekly batches)

Wave 2 was written directly on `feat/prd-251c-w2-weekly-batches` (no Ralph loop), stacked on Wave 1. Run this on the local stack once the branch is merged into local main, the backend is restarted (the `prd251c_wave2` migration runs at boot) and the frontend is rebuilt. Record what failed in the PR.

The defaults this wave took while the open questions are open:

- **O1:** a new plan is weekly. Plans saved before this wave stay daily.
- **O2:** "Approve the week" works without the workspace's series-approval switch. Each post is still approved by its own hash.
- **O3:** the batch is made on Sunday at 17:00.
- **O4:** monthly is offered but is not the default.
- **O8:** notices link to the week's review. There are no approve buttons inside Telegram or Slack.

## 0. Set up

1. **Local main** carries Wave 1 and this branch. Restart the backend, then check the boot log for `prd251c_wave2`. TESTER rebuilds the frontend; never start a build yourself.
2. **Workspace c1** has Socials on, LinkedIn connected, and a plan from before this wave.

## 1. A new plan makes its week at once (US-C201, US-C202)

- [ ] **New plan → Making and approving:** the Rhythm is **Weekly**, the batch is made on **Sunday at 17:00**, and research is set to **Saturday**.
- [ ] **The header line** reads "Every Sunday at 17:00 the next 7 days' posts are made…". Once saved, "The next batch is made Sun …, 17:00." shows under Making.
- [ ] **Move the batch day to Friday:** research follows to Thursday. Set research to Monday yourself, move the batch day again, and research stays on Monday.
- [ ] **Monthly:** pick Monthly, then "The month is made on the 25th". Save and reload, and both are kept.
- [ ] **An existing plan** (from before this wave) still says Daily, and its posts are still made on their day.

## 2. The batch is made (US-C203)

- [ ] **Set the batch day to today, a few minutes ahead,** with 3 to 4 topics in the bank. At that time, the next 7 days' posts are made within a tick or two (5 minutes each).
- [ ] **The Queue** groups them as one week ("Week of …"), not day by day.
- [ ] **A slot no connected channel posts** (a row on a channel you have not connected) is skipped. Its notice says why, and the week is still announced.
- [ ] **"Your week is ready"** arrives once, naming how many posts are in it. Its message carries the Queue's link.

## 3. Approve the week (US-C205)

- [ ] **Approve the week** in the Queue approves every post of the week you were shown, and each is scheduled into its slot. The workspace's series-approval switch stays off.
- [ ] **Edit one post** in another tab after opening the Queue, then press Approve the week. That post is left and listed ("changed since you saw it"), and the rest are approved.
- [ ] **A viewer** sees no Approve button.

## 4. The evening reminder (US-C206)

- [ ] **At 20:00 the evening before** a day with posts still waiting, one notice says "N posts for tomorrow still wait for approval". Nothing comes when tomorrow's posts are approved.

## 5. Research the day before (US-C207)

- [ ] **The Content bank** says when research last ran.

## 6. The review fixes (after the Wave 2 code review)

- [ ] **A new daily plan without a time** (Plan with Auto, or the API) makes its posts at 07:00, not 17:00.
- [ ] **A slot whose time came before it was made** (only if the backend was down across a batch): the week's notice ends "…; N passed before they could be made". The skipped slot shows in the plan's batch record.
