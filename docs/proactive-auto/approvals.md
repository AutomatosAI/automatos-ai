# Approvals

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this page:

- [orchestrator/modules/tools/formatting/approval_view.py](../../orchestrator/modules/tools/formatting/approval_view.py)
- [orchestrator/modules/policy/rule_sentences.py](../../orchestrator/modules/policy/rule_sentences.py)
- [orchestrator/services/approval_views.py](../../orchestrator/services/approval_views.py)
- [orchestrator/api/approval_views.py](../../orchestrator/api/approval_views.py)
- [orchestrator/services/board_approval.py](../../orchestrator/services/board_approval.py)
- [frontend/components/widgets/approval-view-parts.tsx](../../frontend/components/widgets/approval-view-parts.tsx)
- [frontend/lib/theme-sdk/host-cards/approvals.tsx](../../frontend/lib/theme-sdk/host-cards/approvals.tsx)
- [frontend/components/activity/board/blocked-reason.ts](../../frontend/components/activity/board/blocked-reason.ts)
- [frontend/components/activity/board/board-task-viewer.tsx](../../frontend/components/activity/board/board-task-viewer.tsx)

</details>

Anything Auto or an agent wants to do that your workspace's policy says needs a person (sending to other people,
acting in another app, deleting, publishing, or any run under a stricter policy) stops and waits on an approval
card. Proactive Auto makes that card say exactly what will happen and why it is asking.

## The approval card

The card appears in the chat where the work was asked for. There are two kinds:

- **Action approval needed**: one tool call, such as sending a message.
- **Mission plan — approval needed**: a Mission's plan, before any of its steps run.

When the server has the details, the card shows, in this order:

| Part | Example |
|---|---|
| Who asked, and the verb | ASKED BY AUTO · SEND |
| The action | Send the weekly summary to the team channel |
| The target | → #general · Slack |
| A preview | The exact text or content that will go out, as plain text |
| The rule that asked | ⚖ Balanced policy: ask before anything that sends to other people or acts in another app |

The preview is what the call will really send, with the same redaction the card has always used.

### The buttons

| Card | Allow | Refuse |
|---|---|---|
| Action approval | **Approve & run** | **Deny** |
| Mission plan | **Approve & run** | **Reject** |

After **Deny**, the card says "Denied — the action will not run."

With the card focused, the keyboard works too: **1** refuses and **3** approves. **2** does nothing for now: it is kept
for an "Always allow" choice, which is not built. Every approval is for this one action only.

Only owners, admins and editors can approve or refuse an approval; viewers can't.

The approval card is drawn by Automatos itself, whatever theme you use. A theme can place it but can't restyle or
replace it.

## The rule that asked

Every approval names the rule behind it in one plain sentence, in your language. The rules come from your workspace's
policy and its approval settings:

| Rule | Sentence |
|---|---|
| A tool marked for confirmation | "Ask before anything that sends to other people or acts in another app" |
| | "Ask before anything that deletes or can't be undone" |
| | "Ask before anything that publishes" |
| The Balanced policy | "Balanced policy: ask before anything that sends to other people or acts in another app" |
| The Strict policy | "Strict policy: ask before any change in this workspace" |
| The Permissive policy | "Permissive policy: ask only before anything that deletes or can't be undone" |
| Mission and ticket approval: always ask | "Ask before any Mission or task runs" |
| Mission and ticket approval: ask above the budget | "Ask before work that costs more than your approval ceiling" |
| Mission and ticket approval: full autonomy, while it is off | "Ask before work runs while full autonomy is off" |

When the rule can't be named, the card says "Ask before this runs". It never shows a policy's internal name.

## Tickets waiting for approval

A ticket on the Board that is waiting for an approval shows the same sentence. In the ticket, the heading reads
"Waiting for your approval (grant #N)" with the rule under it, and the buttons **Approve #N** and **Move to In
Progress**. Before Proactive Auto, this spot showed the policy's internal key.

## Approving from the phone

Automatos Go opens the same approval view from the card in the thread or from an approval's notification: who asked,
the action, the target, the preview and the rule. While the app's Face ID setting is on, it asks for Face ID before
you approve; refusing never asks. How the sheet looks in each of the app's styles is in the
[Automatos Go repository](https://github.com/AutomatosAI/automatos-go).

An approval that was already answered, revoked or has lapsed still opens, but offers no buttons.

## Questions

**Why does the card say "Approve & run" and not "Allow once"?** The buttons keep their existing words until an
"Always allow" choice exists. That choice would write a standing rule, and it isn't built.

**Does the card show when the approval expires?** Not on the web. The view carries an expiry when there is one, but
the web card doesn't show it.
