# @automatos/theme-sdk — changelog

What each version of the SDK's types added. A `1.x` only adds: a theme built against an older `1.x` keeps working on
a newer host. The host refuses a theme whose `engines.sdk` its version does not meet, so a theme that uses a name
below asks for the version that added it (`^1.5.0` for what 1.5.0 added).

## 1.5.0

Themes talk to individual agents and use Plan. No name is added (fields only), and the host contract stays 1.1.0.

- **chat:** `HostChatController` gains who answers and Plan, as Studio's chat has them:
  - `agentId` (`number | null`): the agent the next message goes to; null is Auto, as every conversation starts.
  - `chooseAgent(id)`: picks an agent of the roster (`pages/agents`' `useAgents`) for this conversation, or Auto with
    null, and returns true. It returns false, changing nothing, for a session agent (`configuration.runtime` 'cli'),
    which is never a chat partner (Studio opens its Claude Code session instead), for an id the roster doesn't have,
    and while the roster loads. Anything but a whole number or null throws a TypeError.
  - `planMode` and `setPlanMode(on)`: Plan (research and a plan, nothing run), the host's own switch, the one
    Studio's Plan button and a `/chat?mode=plan` link set; turning it on turns Mission mode off. Anything but true or
    false throws a TypeError.
  The next message goes with both, and the host's AI label names the agent while it answers.
- **pages/agents:** `RosterAgent.configuration` (`{ runtime?: string | null } | null`, optional), which the roster
  already carried: `runtime` 'cli' marks a session agent.
- **A link, not a name:** `/chat?agent=<id>` (`HostLink`, `useHostRouter`) opens the host's own chat with that agent
  picked, as a pick in Studio's message box: a session agent's Claude Code session opens in the chat's canvas. The
  link keeps the host's chat page under a theme that owns `/chat`.

Fakes that build a `HostChatController` (a harness's sample data) add the four fields.

## 1.4.0

PRD-268 Wave K, for themes that own `/chat` (Play, HUD). No area is added, and the host contract stays 1.1.0.

- **chat:** `HostChatMessage.knownParts`, the message's parts of the types a theme can read, in its order, each
  checked and frozen: `HostTextPart` (`text`), `HostBriefPart` (`brief@1`, the day's Brief), `HostSuggestionPart`
  (`suggestion@1`, a suggestion's id) and `HostReplyChipsPart` (`reply_chips@1`, the chips the host draws), as the
  union `HostMessagePart`. A part of any other type, or one that isn't what its type says, is left out; brief,
  suggestion and chip parts count on Auto's messages only. A theme finds the brief by its part, not by its prose. A
  chip has no route of its own: a tap is the person's message, sent with the controller's `send`.
- **voice:** `BriefVoice` takes `BriefVoiceProps`: `onSpeakingRow(row)` hears the brief row Auto starts reading (its
  index in the brief's rows, in the sections' order: the number Auto says, less one) and null between rows, during the
  greeting and once the reading ends. Its root also carries `data-voice-row="<index>"` while a row is read.
- **proactive:** `useRateBrief(stateHash?)`, the brief's thumbs up or down (`RateBriefInput`: `state_hash`, `rating`
  of `BriefRating`, `up` or `down`; it answers `BriefRated`), through the host's existing digest feedback route, the one
  `cards.BriefCard` posts. It is a `RateBriefAction`: given the brief's `state_hash`, its `rating` is the person's own
  rating of that brief (`up`, `down` or null), read from the server and shared with `BriefCard`'s thumbs, so a press
  in either shows in both. The server keeps one rating per person per brief: a second press changes it.
- **cards:** `BriefCard` takes `thumbs` (default true). Its thumbs show the person's remembered rating; `thumbs={false}`
  draws none, for a theme that draws its own with `useRateBrief` beside the card.
- **proactive (docs):** `useSuggestions` is the signed-in person's own open suggestions, never a row addressed to
  someone else.

Fakes that build a `HostChatMessage` (a harness's sample data) add `knownParts`.

## 1.3.0

PRD-268 (#1375): the `proactive` area (the brief, readiness, suggestions with Start and Hide, calibration), the cards
area's proactive defaults (`BriefCard`, `SuggestionCard`, `StatusLabel`, `ReadinessList`, `CalibrationCard`,
`ReplyChips`), `voice.BriefVoice` and `theme.DISPLAY_STATE_LABELS`.

## 1.2.0

Theme voice (#1334): the `voice` area (`VoiceButton`, `useVoice`, `readVoiceLevels`).

## 1.1.0

PRD-261 W-E3 (#1249): the `storage` area, small per-user theme state through the host.

## 1.0.0

PRD-261 Phase A: the SDK's areas and names (`theme`, `context`, `navigation`, `notices`, `events`, `chat`,
`deliverables`, `cards`, `looks`, `assets` and the `pages/*` data hooks).
