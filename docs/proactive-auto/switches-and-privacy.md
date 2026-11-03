# Switches, editions and privacy

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this page:

- [orchestrator/services/proactive/switches.py](../../orchestrator/services/proactive/switches.py)
- [orchestrator/services/proactive/settings.py](../../orchestrator/services/proactive/settings.py)
- [orchestrator/services/proactive/gates.py](../../orchestrator/services/proactive/gates.py)
- [orchestrator/services/proactive/loop.py](../../orchestrator/services/proactive/loop.py)
- [orchestrator/services/proactive/signals.py](../../orchestrator/services/proactive/signals.py)
- [orchestrator/modules/tools/discovery/check_in_lane.py](../../orchestrator/modules/tools/discovery/check_in_lane.py)
- [orchestrator/core/read_only_call.py](../../orchestrator/core/read_only_call.py)
- [orchestrator/services/push/suggestion_rules.py](../../orchestrator/services/push/suggestion_rules.py)
- [orchestrator/services/calibration/switch.py](../../orchestrator/services/calibration/switch.py)
- [orchestrator/api/auto_calibration.py](../../orchestrator/api/auto_calibration.py)
- [orchestrator/services/credits/routing.py](../../orchestrator/services/credits/routing.py)
- [orchestrator/config.py](../../orchestrator/config.py)

</details>

Proactive Auto ships switched off. This page says what is off, who can turn each part on, how the local and hosted
editions differ, and exactly what a check-in can and can't do with your data.

## Off by default

| What | Default | Who turns it on |
|---|---|---|
| Proactive Auto on the platform (the mode: off, shadow or live) | Off | The platform operator: on the hosted edition, Automatos; on the local edition, whoever runs the install. |
| Proactive Auto in a workspace | Off | The workspace's owners and admins. |
| The suggestion push to the phone | Off, and built | The platform operator. Until then, suggestions still ring the in-app bell. |
| Calibration (Auto saying back what it thinks it knows about your work, on your first chat) | Off, and built | The platform operator. Until then, nothing about it shows. |

A workspace runs Proactive Auto only when **both** the platform mode is shadow or live **and** the workspace has
switched it on. Either one off means off. If Automatos can't read a switch, it treats it as off.

Owners and admins switch the workspace on or off in **Settings → Orchestrator → Proactive Auto**, and tighten its
limits there. Everyone else sees the switch there, read-only. Under the switch, one line says what Auto is doing in the
workspace:

- "Auto is checking in and suggesting." when the platform mode is live;
- "Auto is checking in quietly: suggestions are recorded, but nothing is sent yet." when it is shadow;
- "Switched on here, but Proactive Auto isn't running on this server yet." when the workspace is on but the platform
  mode is off.

The same switch is in the API (see [the engineering routes](engineering/data-and-routes.md#routes)).

### The suggestion push

When it is turned on, a suggestion that matters today can reach your phone as a notification, with these rules:

- it says only "Auto has a suggestion", unless the workspace chose to show suggestion titles;
- at most two a day per person, and never in quiet hours;
- only to a phone whose app can show it;
- each person can switch it off with the other phone notification switches.

### Calibration

When it is turned on, on your first chat in a workspace Auto asks once: "Auto reads the titles of your recent work to
say what it knows." Choose **Go ahead** or **Not now**. If you go ahead, Auto reads only the **titles and kinds** of
your recent Deliverables, your own chats and the workspace's Knowledge you can see, plus the names of installed
skills, never their contents. It then says up to three things it thinks it knows, each with how many items it came
from (never a percentage), and three things it can do.

- **Reject** on a line strikes it through ("I'll forget that") and remembers the correction.
- **Sounds right** remembers every line you didn't reject, privately, as yours.
- **Something's off** sends that as your message, so you can say what's wrong.
- **Not now** reads nothing. You are asked once per workspace.

A line that names something secret is never stored, and the card says so.

## What you control as a person

| Control | What it does |
|---|---|
| **Pause suggestions** (per workspace) | Check-ins make no suggestions for you, and Auto won't promise you follow-ups. Your daily brief still comes. |
| **Quiet hours** (a start time, an end time, your time zone) | No suggestion bell or push in those hours. Check-ins still run and record; what they find waits for your brief. The workspace's own quiet hours apply on top. |
| **Thumbs on the brief** | Your rating of each brief. |
| **Hide** | Puts a suggestion away for 14 days, with an optional reason. |
| **Cancel** on a follow-up | Stops it. |

**Pause suggestions** is in **Settings → Orchestrator → Proactive Auto**, for every member. Quiet hours are set
through the API (`PUT /api/proactive/me`); there is no control for them in Settings yet.

## Local edition and hosted edition

Proactive Auto is the same code in both editions. What differs:

| | Local edition | Hosted edition |
|---|---|---|
| Who sets the platform mode | The person who runs the install | Automatos |
| Who pays for check-ins | Your own model keys | Your workspace's AI credits, never the trial: a workspace with only trial credits left runs no check-ins and no follow-ups |
| A ticket a suggestion's Start files | Your Start is its approval | Follows your workspace's approval policy, so it may wait for an approval |
| "What Auto depends on" details | Also show where each check reached and how long it took | Never show a host or a time |
| Phone notifications | Only after someone agrees to send them through Expo (**Settings → Phones & devices**) | Available |

In both editions, the daily spend ceiling, if your workspace has one, stops new check-ins and Starts once it is
reached. Hide always works.

## Privacy: what a check-in reads, and that it never writes

A check-in is a background run that **only reads**. This is enforced by the server for every call the check-in
makes, not by asking the model nicely.

**What it can read.** Your workspace, through the read-only platform actions: the Board and its tickets, Missions,
Playbooks and their runs, reports, heartbeat results, approvals, documents and the other read actions Auto has. Auto's
own check-in starts from what changed: failed, blocked or stuck tickets, heartbeat errors, missed or benched
Playbooks, reports waiting for approval, approvals about to lapse, missed scheduled tasks, low credits. An agent's
check-in also has the agent's own instructions and what it remembers about its work.

**What it can't reach.**

- Connected apps (through Composio), plugins, workspace files and code tools: never offered, and refused if tried.
- The web: web search and fetching pages are refused in a check-in, so nothing it read leaves the workspace.
- Sub-agents and Missions: a check-in can't start other work.
- Your mail or calendar: there is no such read.

**What it writes.** Only one thing: a suggestion for a person, through the one tool that records one. It cannot
send, edit, delete, file a ticket, schedule anything or start a Mission. Even reads that normally keep a small
record of their own (a search log, a usage count) skip it during a check-in. Anything not positively known to be a
read, including an action Automatos doesn't recognise, is refused.

**What it sends to the model.** The check-in's own prompt, the changes it found, and the results of the reads it
made, quoted as information, never as instructions. It uses the platform's model for Auto (Claude Haiku 5.5 by
default), not the agent's own model, with tight limits per run: at most 4 model calls, 800 output tokens each, about
12,000 tokens of context and $0.05. Past the cost limit the run stops.

**What it suggests.** Each suggestion must cite reads that check-in really made. Its text is shown as plain text, and
Start never posts it as your message.

**Follow-ups** read the same way, plus the reason Auto wrote and the last five messages of the chat they came from.
They are private to the person they were promised to.

**Every run is recorded**: when it ran, what it found, how many suggestions it made, the model, tokens and cost.
Owners and admins can read the workspace's run history through the API. Every Start, Hide, switch change and
follow-up schedule, fire and cancel is in the workspace audit log; a follow-up's reason never is.
