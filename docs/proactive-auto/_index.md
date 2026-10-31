# Proactive Auto

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this page:

- [orchestrator/services/proactive/switches.py](../../orchestrator/services/proactive/switches.py)
- [orchestrator/services/proactive/heartbeat_bridge.py](../../orchestrator/services/proactive/heartbeat_bridge.py)
- [orchestrator/services/proactive/runner.py](../../orchestrator/services/proactive/runner.py)
- [orchestrator/services/proactive/delivery.py](../../orchestrator/services/proactive/delivery.py)
- [orchestrator/api/proactive_brief.py](../../orchestrator/api/proactive_brief.py)
- [orchestrator/modules/tools/discovery/handlers_follow_ups.py](../../orchestrator/modules/tools/discovery/handlers_follow_ups.py)
- [frontend/components/proactive/brief-on-open.tsx](../../frontend/components/proactive/brief-on-open.tsx)
- [frontend/components/proactive/pinned-suggestion.tsx](../../frontend/components/proactive/pinned-suggestion.tsx)
- [frontend/components/proactive/suggested-list.tsx](../../frontend/components/proactive/suggested-list.tsx)

</details>

Proactive Auto is Auto keeping an eye on your workspace while you are away. Auto checks what changed, tells you
what needs you, and suggests a next step when it finds one. It never acts on its own: a suggestion becomes work only
when a person presses **Start**.

Most checks find nothing new, and then Auto says nothing. When it does have something, you see it as a suggestion
with **Start** and **Hide**, and in your daily brief.

This topic is for the people who use a workspace: owners, admins, editors and viewers. For how it is built and run,
read [Proactive Auto: engineering](engineering/_index.md).

## Pages in this topic

| Page | What it covers |
|---|---|
| [The daily brief](daily-brief.md) | The brief waiting in your Auto thread each day: its sections, the thumbs and Listen |
| [Suggestions](suggestions.md) | Where suggestions show, what Start and Hide do, and why one appears |
| [Follow-ups and reply chips](follow-ups-and-reply-chips.md) | "I'll check again at 15:10", Cancel, missed follow-ups, and the short answers under Auto's question |
| [Approvals](approvals.md) | The approval card: what Auto wants to do, to whom, and the rule that asked |
| [Status words and what Auto depends on](status-and-readiness.md) | The glyph and word on every piece of work, and the "What Auto depends on" list |
| [Switches, editions and privacy](switches-and-privacy.md) | What is off by default, who turns things on, local vs hosted, and what a check-in reads |

## What Proactive Auto does

| Feature | What you get |
|---|---|
| Check-ins | Your agents' scheduled heartbeats, and Auto's own workspace check, become read-only check-ins. They look; they never change anything. |
| Suggestions | What a check-in finds, as a card with **Start** and **Hide**. |
| The daily brief | One brief per person per day, posted in your Auto thread: what needs you, what Auto suggests, what's on today and what happened since. |
| Follow-ups | Auto can promise to check again later in your chat, and keeps to it once. |
| Reply chips | When Auto asks you to choose, two or three likely answers appear as buttons. |
| Approvals | The approval card shows the exact action, its target, a preview and the rule that asked for your approval. |
| Status words | Every ticket, Mission and needs-you row says where it stands in one glyph and one word. |
| What Auto depends on | A list in Settings of what Auto needs to work, and anything it couldn't check. |

## The three modes

Proactive Auto has three modes: **off**, **shadow** and **live**. The platform operator picks the mode, and a
workspace owner or admin switches it on for their workspace. Until both are on, your workspace is **off**. See
[Switches, editions and privacy](switches-and-privacy.md) for who turns what on.

| Mode | What runs | What you see |
|---|---|---|
| **Off** (the default) | Heartbeats run as they always have. | No brief on open, no new suggestions and no follow-ups. If you use the phone app's scheduled daily brief, it still arrives. |
| **Shadow** | Heartbeats run as they always have, and a check-in runs beside each one that converts. The check-ins record what they would suggest. | The daily brief appears when you open Automatos. Owners see recorded suggestions in the Command Center's **Suggested** list, in the brief and above the composer in their Auto thread. Nothing is posted in the thread for them, and no bell rings. Auto does not offer follow-ups. |
| **Live** | Check-ins run **instead of** the heartbeats that convert. Those heartbeats no longer file a ticket or a report on every run. | Everything in shadow, plus: new suggestions are posted in your Auto thread, the bell rings for one that matters today, and Auto can schedule follow-ups. |

Which heartbeats convert:

- **An agent's heartbeat** converts unless it is set to act. A heartbeat that was already allowed to act on its own
  (its "auto act" setting) keeps acting, as before. Claude Code agents keep their current heartbeat.
- **Auto's own heartbeat** (Settings → Orchestrator → Enable Heartbeat) converts when its Proactive Behavior Level is
  **Silent** or **Notify Only**. At **Act & Notify** or **Fully Autonomous** it keeps working as before. At
  **Silent** the check-in never rings the bell.

In live mode, an agent's heartbeat rings the bell only when its check-in made a suggestion, or when it started to
fail. A heartbeat that keeps failing gives you one card, not one per run (see [Suggestions](suggestions.md)).

## Who suggestions are for

A check-in's suggestions go to the **workspace owner**. Other members don't get them, though an owner or admin can
see and hide every open suggestion in the workspace from the Command Center.

A follow-up is the exception: when Auto checks again for you and finds something, its suggestion is addressed to
you, the person who was promised the follow-up.

## On your phone

Automatos Go, the phone app, shows the same brief, your suggestions and the approval view. While its Face ID setting
is on, it asks for Face ID before you approve something or start a suggestion; refusing and hiding never ask. How
each looks in the app's styles, and how to set it up, is
documented in the [Automatos Go repository](https://github.com/AutomatosAI/automatos-go). The suggestion push to the
phone is built but switched off; see [Switches, editions and privacy](switches-and-privacy.md).

## Related

- [Heartbeat & Proactive Assistant](../heartbeat-proactive-assistant/_index.md): the heartbeat scheduler that check-ins
  run on.
- [Themes](../themes/_index.md): installed themes show the brief, suggestions and approvals in their own style.
