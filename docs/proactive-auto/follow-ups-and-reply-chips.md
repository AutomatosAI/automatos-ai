# Follow-ups and reply chips

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this page:

- [orchestrator/modules/tools/discovery/handlers_follow_ups.py](../../orchestrator/modules/tools/discovery/handlers_follow_ups.py)
- [orchestrator/services/proactive/follow_ups.py](../../orchestrator/services/proactive/follow_ups.py)
- [orchestrator/services/proactive/follow_up_fire.py](../../orchestrator/services/proactive/follow_up_fire.py)
- [orchestrator/services/proactive/follow_up_strings.py](../../orchestrator/services/proactive/follow_up_strings.py)
- [orchestrator/api/proactive_follow_ups.py](../../orchestrator/api/proactive_follow_ups.py)
- [orchestrator/modules/tools/discovery/handlers_replies.py](../../orchestrator/modules/tools/discovery/handlers_replies.py)
- [orchestrator/modules/tools/execution/reply_chips_turn.py](../../orchestrator/modules/tools/execution/reply_chips_turn.py)
- [frontend/components/proactive/follow-up-line.tsx](../../frontend/components/proactive/follow-up-line.tsx)
- [frontend/components/proactive/reply-chips.tsx](../../frontend/components/proactive/reply-chips.tsx)
- [frontend/components/proactive/thread-reply-chips.tsx](../../frontend/components/proactive/thread-reply-chips.tsx)

</details>

## Follow-ups

In a chat, Auto can say "I'll check again at 15:10" and keep to it. That promise is a **follow-up**: Auto looks
again once, at that time, and tells you only if there is something to say.

### What you see

Under the message where Auto promised it, a line shows the follow-up in your local time:

| Line | What it means |
|---|---|
| **Follow-up at 15:10 · Cancel** | Waiting. **Cancel** stops it. |
| **Cancelled** | You (or Auto, at your request) cancelled it. |
| **Followed up at 15:10** | It ran. |
| **Missed follow-up at 15:10** | Its time passed and it never ran (for example, the server was restarting). It won't run late, and there is nothing to cancel. |

The line is yours only. Nobody else in the workspace sees your follow-ups, and they don't appear on the Command
Center calendar or in the scheduled tasks list. You can also ask Auto in the chat which follow-ups you have, or ask it
to cancel one.

### When it comes

At the time Auto set, it looks again at what the follow-up was about, with the last few messages of your chat. It can
only read, as a check-in does. Then one of three things happens:

- **Nothing changed.** Auto posts nothing.
- **Something to say.** Auto posts **one** message in the chat where it promised, or in your Auto thread if that chat
  is no longer yours. The bell tells you Auto messaged you. The message is plain text: no links, images or code.
- **Something to do.** Auto records a suggestion addressed to you, with Start and Hide (see
  [Suggestions](suggestions.md)). A follow-up never does the work itself.

A follow-up runs once. If it fails, it is not tried again, and it never schedules another follow-up.

### Limits

| Limit | Value |
|---|---|
| Earliest | 10 minutes from now |
| Latest | 7 days from now |
| Waiting at once | 5 per person |
| Scheduled in 24 hours | 20 per person, 100 per workspace |

When a limit is reached, Auto is told so and doesn't promise.

### When Auto won't promise a follow-up

Auto offers follow-ups only in your own chat with it, and refuses to schedule one when:

- your workspace's mode is not **live** (see [the modes](_index.md#the-three-modes));
- you paused your suggestions in this workspace;
- the workspace can't run background work right now (on the hosted edition, when only trial credits are left);
- the follow-up is about a Mission, a Playbook run or a ticket. Those already have a watch that follows the work to
  its result and posts back, so Auto uses that instead.

Each follow-up that runs counts as one ask against your workspace's AI usage, and it is never paid from trial
credits. If the workspace reached its daily spend ceiling, or you paused suggestions in the meantime, the follow-up
does not run.

## Reply chips

When Auto asks you to choose, two or three likely answers appear as small buttons under its message, such as
**Window**, **Aisle**, **Either**. Tap one and its text is sent as your own message. You can always type your own
answer instead.

- Chips appear only under the newest message in the chat, and disappear once a newer message shows.
- Each chip is one to three words, at most 24 characters. There are never more than three.
- Auto offers chips only when it asks you to choose. It never offers them with the daily brief, in a check-in or in a
  follow-up.
- After a tap the chips are disabled. If nothing was sent (for example, a reply was still streaming), they come back.
- On a spoken turn, the chips are shown but never read aloud.
- Chips don't appear in a read-only chat.

Reply chips are Auto's words, so they are marked as generated, and each chip is announced to screen readers as
"Reply: Window".

Offering chips costs nothing extra: Auto's reply ends with the chips, with no second call to the model.

## Questions

**Can my colleagues see that Auto will check back on something for me?** No. A follow-up is private to the person
whose chat scheduled it. The audit log records that a follow-up was scheduled, fired or cancelled, but never its
reason.

**I closed the tab. Does the follow-up still happen?** Yes. It runs on the server at its time, whether or not you
have Automatos open.

**Does a follow-up see my whole chat?** It reads the reason Auto wrote and the last five messages of the chat, and
whatever it can read in the workspace. It can't change anything.
