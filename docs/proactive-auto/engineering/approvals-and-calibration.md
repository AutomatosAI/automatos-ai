# Approvals and calibration

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this page:

- [orchestrator/modules/tools/formatting/approval_view.py](../../../orchestrator/modules/tools/formatting/approval_view.py)
- [orchestrator/modules/tools/formatting/card_digest.py](../../../orchestrator/modules/tools/formatting/card_digest.py)
- [orchestrator/modules/policy/rule_sentences.py](../../../orchestrator/modules/policy/rule_sentences.py)
- [orchestrator/modules/tools/discovery/mission_approval_rule.py](../../../orchestrator/modules/tools/discovery/mission_approval_rule.py)
- [orchestrator/services/approval_views.py](../../../orchestrator/services/approval_views.py)
- [orchestrator/api/approval_views.py](../../../orchestrator/api/approval_views.py)
- [orchestrator/services/board_approval.py](../../../orchestrator/services/board_approval.py)
- [frontend/lib/theme-sdk/host-cards/approvals.tsx](../../../frontend/lib/theme-sdk/host-cards/approvals.tsx)
- [frontend/components/widgets/approval-view-parts.tsx](../../../frontend/components/widgets/approval-view-parts.tsx)
- [frontend/components/activity/board/blocked-reason.ts](../../../frontend/components/activity/board/blocked-reason.ts)
- [orchestrator/api/auto_calibration.py](../../../orchestrator/api/auto_calibration.py)
- [orchestrator/services/calibration/switch.py](../../../orchestrator/services/calibration/switch.py)
- [orchestrator/services/calibration/claim.py](../../../orchestrator/services/calibration/claim.py)
- [orchestrator/services/calibration/sources.py](../../../orchestrator/services/calibration/sources.py)
- [orchestrator/services/calibration/build.py](../../../orchestrator/services/calibration/build.py)
- [orchestrator/services/calibration/answers.py](../../../orchestrator/services/calibration/answers.py)
- [orchestrator/services/calibration/memory.py](../../../orchestrator/services/calibration/memory.py)
- [frontend/hooks/use-calibration.ts](../../../frontend/hooks/use-calibration.ts)

</details>

## `ApprovalView@1`

What an approval card shows: the exact action, its target, its preview and the rule that asked.

```ts
type ApprovalView = {
  grant_id: string                                   // a grant's id, or a Mission plan's (Mission) id
  asked_by: { kind: 'auto' | 'agent'; name: string }
  action: { verb: 'send' | 'pay' | 'sign' | 'delete' | 'publish' | 'run' | string; summary: string }
  target?: { label: string; detail?: string }
  preview?: { kind: 'text' | 'markdown' | 'diff'; body: string }
  rule: { id: string; sentence: string }
  answers: Array<'refuse' | 'allow_once'>           // never 'always_allow'
  expires_at?: string
}
```

It is built in two places (`modules/tools/formatting/approval_view.py`):

- **On the card.** `carries_the_approval_view`, a decorator beside `@carries_the_question` on
  `ToolResultFormatter.format_for_frontend`, adds `approval_view` to the chat's `tool_approval` and `mission_approval`
  payloads. A card's view is in English (the formatter knows no language).
- **From a grant row**, behind `GET /api/v1/approval-grants/{grant_id}/view` (`services/approval_views.py`), in the
  caller's language.

The target and preview are read from the stored call's arguments through the card's own digest
(`card_digest.params_digest`: `_` plumbing dropped, each value cut), never the raw params, so the view shows nothing
the card doesn't. Ticket parameters show by board number.

### The view route

`grant_id` is a number (an approval grant) or a UUID (a Mission plan, which has no grant row). Who may read it is who
may answer it: owners, admins, editors or the super admin (`require_workspace_answerer`); a viewer is 403. Another
workspace's id, a question (answered, never allowed) and an id that names nothing are 404. A database that can't be
read is 503.

- A **grant** still waiting offers `refuse` and `allow_once`; one decided, revoked or past `expires_at` is still shown,
  with no answers. Answers go to the existing `POST /api/v1/approval-grants/{grant_id}/grant` and `…/deny`.
- A **Mission plan** shows the goal, the card's number and the first plan steps. While awaiting approval it names the
  policy holding it (`mission_approval_rule.policy_holding`) and offers both answers, posted to `POST
  /api/missions/{mission_id}/approve` and `/reject`. A plan no longer waiting offers none.

Discovery lists the route as `approval_view`, so the phone opens a view from a card or a push.

### Rule sentences

`modules/policy/rule_sentences.py` gives each rule an approval can come from one sentence, in English and Portuguese
(Brazil). `rule_sentence` never returns machine text; an unknown id reads `fallback` ("Ask before this runs").

| Rule id | Source |
|---|---|
| `confirm.<risk class>` | The confirmation gate's card (`modules/tools/discovery/confirmation_gate.py`), by `classify_action`'s risk class |
| `posture.<posture>.<risk class>` | Every posture × risk-class pair that `PolicyDocument.route_for` sends to ask |
| `approval_policy.<policy>` | The Mission and board approval policy (`core/services/approval_policy.py`): `always_ask`, `auto_below_budget` over the ceiling, `full_auto` without the autonomy gate |

`services/board_approval.py` now writes the sentence into a blocked ticket's reason instead of the raw policy key.
A grant's view finds the policy from that reason (or the older `under '<policy>' policy` text), else the workspace's
policy now. `frontend/components/activity/board/blocked-reason.ts` shows it.

### The cards

`ToolApproval` and `MissionApproval` (`frontend/lib/theme-sdk/host-cards/approvals.tsx`) draw the view through
`frontend/components/widgets/approval-view-parts.tsx`: who asked and the verb, the summary, `→ target · detail`, the
preview as plain text, the rule sentence. They stay protected host surfaces.

- Buttons keep their words: "Approve & run" / "Deny" on the tool card, "Approve & run" / "Reject" on the Mission card.
- Keys, only while the card has focus (never a document listener): `1` refuses, `3` allows once, `2` does nothing
  until Always allow exists.
- `expires_at` is parsed and not drawn.
- **Always allow is not built**: no route, button, confirm sheet or policy write (W2 K10, §12-Q17).

## Calibration (ships off)

On a person's first chat in a workspace, Auto asks once for consent, then says back what it thinks it knows from the
**titles and kinds** of their recent work, and three things it can do. It ships **off** behind `CALIBRATION_ENABLED`
(`services/calibration/switch.py`, read on every call) until consent to read those titles (§12-Q22) is decided. It
does not read the proactive mode.

While off: `GET /api/auto/calibration` answers `available: false` and reads no record, every POST is 404
`not_available`, and discovery doesn't list `calibration`. The web shows nothing.

### When on

1. **Offer.** `GET /api/auto/calibration` returns `{available, offer, consent_line, status, calibration,
   rejected_lines, accepted_lines}`. `offer` is true until the person was offered it (or a claimant died before
   posting). The consent line: "Auto reads the titles of your recent work to say what it knows."
2. **Claim.** `POST /api/auto/calibration {consent, chat_id?, locale?}` claims once per person and workspace with
   `INSERT … ON CONFLICT DO NOTHING` on `proactive_calibrations`' unique `(workspace_id, user_id)`. A second POST is
   409 `already_offered`. `consent: false` records `declined` and reads nothing.
3. **Read** (`services/calibration/sources.py`), at most a few rows per kind, newest first, columns named explicitly:
   Deliverables (title and `artifact_type`, heartbeat reports and archived rows left out), Knowledge documents (file
   name and type, shared or the person's own), the person's own chats (title; not Auto's thread, not a widget
   visitor's, not "New Chat"), and installed, active skills (name and category). No body, summary, chunk, path or
   file. `tests/test_prd268w3_calibration_build.py` spies on the statements.
4. **Build.** One model call at most (purpose `calibration`, a customer purpose: the person's own action). At most 3
   lines, each citing only sources it was given (others dropped), each with its source count, never a percentage (a
   line carrying one is dropped). Three offers name installed skills, else the strings' starters. Nothing to read:
   `empty: true`, three starters, no model call.
5. **Post.** One message `[text, calibration@1]` into `chat_id` when it is the caller's own chat, else their Auto
   thread; source `{origin: 'calibration', calibration_id}`; never announced. A build or post that fails is 503
   `calibration_unavailable`, and the person is offered again.

### Answers and memory

`POST /api/auto/calibration/{calibration_id}/lines/{line_id}/reject` and `POST …/{calibration_id}/accept` (Sounds
right), through the existing memory service (`services/calibration/memory.py`):

- Accept stores each line not rejected as `fact_type` `user_fact`, owner `user:<id>` (private), `source_type`
  `platform_verified`, extra `{origin: 'calibration', calibration_id, line_id}`.
- Reject stores one correction, "Not so: <the line>", with `correction: true`.
- Every content first passes `write_contract.violates_exclusions`: a line that names a secret is never stored, and the
  answer lists it in `not_stored` with a `note`.
- Each answer is idempotent; rejecting after the calibration was answered is 409 `calibration_answered`; a memory
  write that fails is 503 `memory_unavailable` and changes nothing.

"Something's off" is the client sending that text as the person's message, as a reply chip does.
