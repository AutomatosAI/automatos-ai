# Suggestions

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this page:

- [orchestrator/api/proactive.py](../../orchestrator/api/proactive.py)
- [orchestrator/services/proactive/decisions.py](../../orchestrator/services/proactive/decisions.py)
- [orchestrator/services/proactive/decision_text.py](../../orchestrator/services/proactive/decision_text.py)
- [orchestrator/services/proactive/start_ticket.py](../../orchestrator/services/proactive/start_ticket.py)
- [orchestrator/services/proactive/delivery.py](../../orchestrator/services/proactive/delivery.py)
- [orchestrator/services/proactive/delivery_text.py](../../orchestrator/services/proactive/delivery_text.py)
- [orchestrator/services/proactive/failing_streaks.py](../../orchestrator/services/proactive/failing_streaks.py)
- [orchestrator/services/proactive/streak_text.py](../../orchestrator/services/proactive/streak_text.py)
- [orchestrator/services/proactive/expiry.py](../../orchestrator/services/proactive/expiry.py)
- [orchestrator/modules/proactive/dedupe.py](../../orchestrator/modules/proactive/dedupe.py)
- [frontend/components/proactive/suggestion-card.tsx](../../frontend/components/proactive/suggestion-card.tsx)
- [frontend/components/proactive/suggestion-parts.tsx](../../frontend/components/proactive/suggestion-parts.tsx)
- [frontend/components/proactive/pinned-suggestion.tsx](../../frontend/components/proactive/pinned-suggestion.tsx)
- [frontend/components/proactive/suggested-list.tsx](../../frontend/components/proactive/suggested-list.tsx)
- [frontend/components/proactive/thread-cards.tsx](../../frontend/components/proactive/thread-cards.tsx)
- [frontend/components/activity/board/from-suggestions.ts](../../frontend/components/activity/board/from-suggestions.ts)

</details>

A suggestion is something a check-in found that might need doing, with what Auto would do about it. You decide:
**Start** turns it into work, **Hide** puts it away. Nothing happens until you press one.

## Why a suggestion appears

A check-in looks at what changed since its last successful run (or the last 24 hours) and suggests something only
when it found something new. Auto's own check-in looks at things like:

- a ticket that failed, got blocked or is stuck waiting for a person;
- a heartbeat that errored or missed its objective;
- a scheduled Playbook that missed its run, was benched, expired or ran out of corrective budget;
- a report that asks for approval and nobody has acknowledged;
- an approval about to lapse in the next 24 hours;
- a one-off scheduled task whose time passed without it running;
- AI credits running low while the balance is moving.

When none of these changed, Auto's check-in ends there, without asking the model anything. An agent's check-in runs
the agent's own heartbeat instructions, but can only read.

Every suggestion names what it read to reach it (up to five references: a ticket number, a report, a Mission, a
chat), and only things that check-in really read.

Auto does not repeat itself. A suggestion is not made again while the same one is open, or for 14 days after you hid
it.

### When an agent keeps failing

If an agent's check-in fails (for example, its AI provider refuses its key), you get **one** card for the whole
streak, not one per run, such as "Scout's check-ins fail: the AI provider refuses its key (HTTP 401)". The card says
when the failures started and how many there have been. No ticket is filed.

- From the third failure in a row, the agent checks in less often: its interval doubles each time, up to once a day.
  The card says how often it checks in now.
- **Start** on that card gives Auto a ticket to find out why and fix it.
- When a check-in works again, the card expires and says so ("…'s check-ins work again, after 4 failures in a row").
- If you hide a streak's card, that streak gets no other card.

## Where suggestions show

| Place | What you see |
|---|---|
| **Your Auto thread** | Each new suggestion is posted as an Auto message with its card, in live mode. |
| **Above the composer in your Auto thread** | Your newest open suggestion, on one line with Start and Hide. Click its title to open the full card; Escape folds it back. It steps aside while a Mission suggestion is waiting in the chat. |
| **Command Center → Summary → Suggested** | The open suggestions beside Auto's read, each with "What Auto read" links to the tickets, reports or chats it cites. Owners and admins see every open suggestion in the workspace here; anyone else sees their own. |
| **The daily brief** | Up to three of your open suggestions, in its **Suggested** section. |
| **The bell** | "Auto has a suggestion: 1 suggestion for today". It links to the card on the Command Center. |

The bell rings only in live mode, and only when all of these hold:

- the run made a suggestion that matters **today** (Auto marks each one "today" or "can wait");
- you are not in your own quiet hours, or the workspace's;
- you have had fewer than two suggestion bells today.

The bell rings at most once per check-in, however many suggestions it made. Suggestions that can wait only appear in
the places above, without a bell.

## The card

| Part | Example |
|---|---|
| Label | **Auto suggests** |
| Title and detail | What Auto found, in its own words |
| What Start does | "Start does: files a ticket on the Board for an agent to work." or "Start does: opens a chat with Auto about this." |
| Buttons | **Start** and **Hide** |
| Caption | "Auto is an AI", once per card |

The card's title and detail are Auto's words, written after reading your workspace. They are shown as plain text, and
Start never sends them to anyone as your message.

## Start

**Start** makes the suggestion's work. Who can press it: owners, admins and editors. Viewers can't.

- **A ticket suggestion** files a ticket on the Board, as if you had filed it: assigned to the agent the suggestion
  named, or left in the Inbox when it named none. The ticket waits in **Review** for a person when the work is done.
  Its brief quotes the suggestion and what Auto read, as information, and says your Start is the instruction. The
  ticket is put under watch, so its outcome is followed through.
- **A chat suggestion** is marked started. On the web the card says so and nothing else opens; talk it through with
  Auto in your Auto thread.

After Start, the card says "Started. Auto is on it." for a few seconds, then folds to one line: "{title} · Started".

On the local edition, your Start is the approval for the ticket it files. On the hosted edition, the ticket follows
your workspace's approval policy, so it may wait for an approval first, like any other ticket.

### When Start is refused

The card stays where it is and says why:

| Message | Why |
|---|---|
| "Not started: the workspace has reached today's spend ceiling, so no new work starts until the ceiling is raised or the day's window rolls over. The suggestion stays open." | Your workspace's daily spend ceiling was reached. |
| "This suggestion was already started." | Someone started it first. Only one Start ever creates work. |
| "This suggestion was hidden, so it can't be started." | Someone hid it first. |
| "This suggestion has expired." | It was open for more than 7 days. |
| "Starting a Mission or a Playbook run from a suggestion is not available in this version." | Check-ins don't propose these yet. |

## Hide

**Hide** puts a suggestion away. It is never blocked by a switch, a cap or the spend ceiling.

When you press Hide, a short menu asks "Why hide it?":

| Choice | Meaning |
|---|---|
| **Not now** | Right idea, wrong time |
| **Not useful** | Not worth doing |
| **Wrong agent** | The work is for someone else |
| **Just hide** | No reason given |

The reason is recorded with the suggestion. A hidden suggestion folds to "{title} · Hidden", and the same thing is
not suggested again for 14 days.

Who can hide: the person the suggestion is for, or an owner or admin. Viewers can't.

## How long a suggestion stays

- An open suggestion expires after 7 days and folds to "· Expired". A failing streak's card stays open while the
  streak goes on.
- You have at most 20 open suggestions. Past that, the oldest expire to make room.
- An agent has at most 3 open suggestions at once, and a check-in makes at most 3.
- A person gets at most 10 new suggestions a day.

Your workspace's owners and admins can lower these limits; see [Switches, editions and privacy](switches-and-privacy.md).

## Following started work

On the Board, the **From suggestions** toggle (in the board's filters) shows only the tickets that a suggestion's
Start filed. Your brief's **Since your last brief** section says what each started suggestion became.

## Questions

**Why do I never get suggestions?** Check-in suggestions go to the workspace owner. You get them only from your own
follow-ups (see [Follow-ups and reply chips](follow-ups-and-reply-chips.md)), or if you are the owner.

**Can I stop suggestions for a while?** Each person can pause their own suggestions in a workspace, and set their
quiet hours. Today these are set through the API, not a switch in Settings; ask your administrator, or see the
[engineering routes](engineering/data-and-routes.md#routes).

**Does the suggestion text ever run by itself?** No. A suggestion is text and references. Work starts only on Start,
and Start shows what it will do first.
