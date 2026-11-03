# Status words and what Auto depends on

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this page:

- [orchestrator/services/proactive/display_state.py](../../orchestrator/services/proactive/display_state.py)
- [orchestrator/services/proactive/strings.py](../../orchestrator/services/proactive/strings.py)
- [frontend/lib/proactive/display-state.ts](../../frontend/lib/proactive/display-state.ts)
- [frontend/components/proactive/status-label.tsx](../../frontend/components/proactive/status-label.tsx)
- [frontend/components/proactive/use-row-retry.ts](../../frontend/components/proactive/use-row-retry.ts)
- [orchestrator/services/readiness/items.py](../../orchestrator/services/readiness/items.py)
- [orchestrator/services/readiness/service.py](../../orchestrator/services/readiness/service.py)
- [orchestrator/services/readiness/strings.py](../../orchestrator/services/readiness/strings.py)
- [frontend/components/settings/what-auto-depends-on.tsx](../../frontend/components/settings/what-auto-depends-on.tsx)
- [frontend/components/proactive/readiness-list.tsx](../../frontend/components/proactive/readiness-list.tsx)

</details>

Two things tell you where you stand at a glance: one set of **status words** on every piece of work, and the
**What Auto depends on** list, which says what Auto needs and what it couldn't check.

## Status words

Tickets, Missions and needs-you rows all say where they stand the same way: a glyph and a word, never a colour alone.
The server decides the state, so every screen, theme and the phone agree.

| Glyph and word | Means |
|---|---|
| ◆ **Needs you** | A person must act: a review, a question, an approval or a stuck ticket. |
| ● **Working** | In motion: a ticket Assigned or In progress, a Mission planning, running or checking its work. |
| ✓ **Ready to look at** | Finished and waiting for you to look (used for started suggestions in the brief). |
| ✓ **Done** | Finished: a ticket Done or Closed, a completed Mission, a granted approval. |
| ✕ **Couldn't finish** | Failed. |

Some rows show nothing: an Inbox ticket, a Blocked one, a cancelled or paused item. When the state isn't known, the
row shows no word rather than a guess.

### The next step

After the word, a line starting with ▸ says what happens next, written by the server in your language: for a stuck
ticket, what it's waiting for; for an approval, what you are being asked to allow; for a failure, the first sentence
of the error.

### Try again

A failed row can offer **Try again**:

- a failed ticket runs again ("Sent to run again");
- a failed Mission is replanned, keeping its finished steps ("Mission replanned — new tasks generated").

### Where they show

- the needs-you widget on the Command Center, with Needs you rows first;
- the Missions list, in Studio and Classic;
- board cards, in both shells;
- the rows of the daily brief.

Themes show the same words. A HUD-style theme may show them in capitals ("NEEDS YOU"), with the same glyphs.

## What Auto depends on

**Settings → Orchestrator → What Auto depends on** lists what Auto needs to work and whether each is ready. It is the
same in every theme, because Settings always belongs to Automatos.

| Item | What it checks | Fix it offers |
|---|---|---|
| **Automatos server** | The server and its database answer. | none |
| **Auto's model** | A model is chosen for Auto, and it has a key to run with. | "Choose Auto's model", "Add a key" |
| **Web search** | A search engine is set up for Auto's web search. | none |
| **Phone notifications** | Notifications can reach your phone. On the local edition, whether someone agreed to send them through Expo. | "Answer in Phones & devices" |
| **Microphone** | This browser can capture audio, for an Auto Live call. | none (checked by the browser) |
| **Speech** | This browser can recognise and speak, for hold-to-talk. | none (checked by the browser) |

Each item shows one of three states:

| State | Means |
|---|---|
| ✓ **Ready** | It works. |
| ○ **Not checked** | The check didn't answer in time ("Didn't answer in time."). This never stands in for a failure the server knows about. |
| ◆ **Needs setup** | A person must do something. The fix, when there is one, is a link to the right place in Settings. |

- **Check again** runs every check now. A workspace can check three times a minute; past that it says "Checked a
  moment ago". Between presses, the list shows the last check.
- Each check has a few seconds to answer before it reads **Not checked**.
- On the **local edition**, each item's detail also says where the check reached and how long it took (for example
  `localhost:5432 · 12 ms`). The **hosted edition** never shows a host or a time.

### In the brief

The daily brief's **Couldn't check** section lists the items the brief depends on (the Automatos server and Auto's
model) that aren't ready. It is never left out, so a quiet brief can't hide a broken connection.

## Questions

**Why does Web search say "Needs setup"?** No search engine is configured for this install, or web search is
switched off on this server. An administrator sets it up.

**Phone notifications says it's waiting for an answer. What answer?** On a local install, someone with access to
**Settings → Phones & devices** has to agree that notifications may be sent through Expo's push service. Until then,
no notification reaches a phone.
