# The daily brief

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this page:

- [orchestrator/api/proactive_brief.py](../../orchestrator/api/proactive_brief.py)
- [orchestrator/services/brief/build.py](../../orchestrator/services/brief/build.py)
- [orchestrator/services/brief/claim.py](../../orchestrator/services/brief/claim.py)
- [orchestrator/services/brief/strings.py](../../orchestrator/services/brief/strings.py)
- [orchestrator/services/brief/needs.py](../../orchestrator/services/brief/needs.py)
- [orchestrator/services/brief/suggested.py](../../orchestrator/services/brief/suggested.py)
- [orchestrator/services/brief/today.py](../../orchestrator/services/brief/today.py)
- [orchestrator/services/brief/since.py](../../orchestrator/services/brief/since.py)
- [orchestrator/services/brief/unchecked.py](../../orchestrator/services/brief/unchecked.py)
- [orchestrator/services/digest_feedback.py](../../orchestrator/services/digest_feedback.py)
- [orchestrator/services/mobile/digest.py](../../orchestrator/services/mobile/digest.py)
- [frontend/components/proactive/brief-on-open.tsx](../../frontend/components/proactive/brief-on-open.tsx)
- [frontend/components/proactive/brief-card.tsx](../../frontend/components/proactive/brief-card.tsx)
- [frontend/components/proactive/brief-sections.tsx](../../frontend/components/proactive/brief-sections.tsx)
- [frontend/components/voice/brief-voice.tsx](../../frontend/components/voice/brief-voice.tsx)
- [frontend/components/activity/autos-read.tsx](../../frontend/components/activity/autos-read.tsx)

</details>

The brief is one message a day from Auto, in your Auto thread, that says what needs you, what Auto suggests, what's
on today and what happened since your last brief. Each person gets their own, and only one per day.

## When it appears

The first time you open Automatos on a given day (your local date), the web app asks for the day's brief. If nobody
has made it yet, Auto writes it and posts it in your Auto thread. If it was already made (on another tab, another
device, or by the phone's scheduled brief), you get that same message. You never get two.

- **The brief on open needs Proactive Auto.** It appears only when your workspace's mode is shadow or live (see
  [the modes](_index.md#the-three-modes)). When the mode is off, opening the app makes no brief.
- **The phone's scheduled brief is separate.** If you use Automatos Go with the daily brief push on, the brief is also
  posted at the time you chose on the phone. That schedule works whatever the mode is. It is
  the same single brief: whichever comes first, the phone's schedule or your first open, makes it.
- **A brief that fails is tried again.** If Auto could not make the brief, the day is not used up: the next time you
  open the app, it tries again.
- **Opening the app is your own action**, so a brief posted on open does not ring the bell. The scheduled brief does,
  as it did before.

The brief is written in your language when the app sends it: English or Portuguese (Brazil), else English.

## Where you see it

| Place | What shows |
|---|---|
| Your **Auto thread** | The full brief card under the brief message: date, greeting, Auto's Read, the sections, the thumbs and Listen. |
| **Command Center → Summary**, in **Auto's read** | The same sections under Auto's Read, while the card is expanded. No thumbs or Listen here. |

Installed themes (Play, HUD and others) can place the brief in their own way; see [Themes](../themes/_index.md).

## What the brief says

The card has the date, a greeting ("Morning, Sam.", "Afternoon.", "Evening." by your local time), Auto's Read (the
same summary as on the Command Center), then five sections, in this order:

| Section | What it lists |
|---|---|
| **Needs you** | The same items the needs-you badge counts: tickets waiting for review, questions and approvals waiting for an answer (Mission plans included), stuck tickets and failed ones. Its count always equals the badge. |
| **Suggested** | Up to three of your open suggestions, the ones that matter today first, then the newest. The count is all your open suggestions. |
| **Today** | What's scheduled for your local day: an agent's routine, a scheduled Playbook, a scheduled task, a Mission's or ticket's deadline, a social post. It is the Command Center calendar, not your personal calendar. |
| **Since your last brief** | What your started suggestions became, plus tickets finished and Missions completed since your last brief. |
| **Couldn't check** | Anything the brief depends on that isn't ready: the Automatos server or Auto's model. It is never hidden, so a quiet brief can't hide a broken connection. |

Each row has a title and a short reason ("An agent's routine runs.", "Started from a suggestion. It couldn't
finish."), and most rows show a status word (see [Status words](status-and-readiness.md)).

- A section shows five rows and folds the rest behind "Show N more". A section with exactly six rows shows all six.
- Every heading except **Couldn't check** shows its count.
- When nothing needs you and nothing is scheduled, the brief says so: "Nothing new needs you, and nothing is
  scheduled today."
- A **viewer's** brief lists no questions and no approvals in **Needs you**, since a viewer can't answer them. The
  count still matches the badge.

On the web, brief rows are for reading. To act on a suggestion, use its card (see [Suggestions](suggestions.md)); to
answer a question or an approval, open it from **Needs you** on the Command Center. A theme that draws the brief
itself can add actions to the rows.

## The thumbs

Under the brief, **Helpful** and **Not helpful** (the group is labelled "Rate this brief") tell Auto whether the
brief was useful. They post the same rating as the thumbs on Auto's Read on the Command Center.

- You have one rating per brief. Once you've rated it, the card shows your rating and the thumbs are disabled.
- Each person's rating is their own: another member's rating doesn't change yours.

## Listen

**Listen** reads the brief aloud: the greeting and the numbered titles of the rows, not the reasons.

- It uses Auto's server voice when the server has one, and the browser's own speech when it doesn't.
- It never plays by itself. Nothing in Automatos speaks first.
- While it reads, the button shows **Stop** and the "AI voice assistant" caption.
- It is disabled while a voice call or a hold-to-talk is running.
- It is on the card in your Auto thread only.

## Asking Auto for the brief

In any chat, you can ask Auto "brief me", "what needs me?" or "what's on today". Auto answers from the same brief,
and by voice it reads the titles only. Asking this way never posts a second brief message.

## Questions

**Why didn't I get a brief today?** Your workspace's mode is off and you don't use the phone's scheduled brief, or
you haven't opened Automatos yet today. An administrator can tell you the mode; see
[Switches, editions and privacy](switches-and-privacy.md).

**Why does the Suggested section show suggestions but my thread has no suggestion messages?** In shadow mode,
suggestions are recorded and listed but not posted in your thread. See [the modes](_index.md#the-three-modes).

**Is the brief made by an AI?** Yes. The card shows "Auto is an AI" once, and Auto's words on it are marked as
generated.
