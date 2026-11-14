# Voice-latency eval (PRD-258 US-004)

Times an Auto Live turn the same way every time, without a person on a call.
The runner speaks Retell's custom-LLM socket protocol to the backend it runs
beside, then reads back the `voice_turns` rows the backend wrote for each turn
(PRD-258 US-001).

The ask set and the results are the owner's. They live in
`~/.automatos-analyst/voice-eval/` (`asks.json`, `results.md`), like PRD-256's
auto-eval, and never in this repo: `.gitignore` here ignores `results.md` and
`asks*.json`. The runner never writes a file: it reads the asks on stdin and
prints to stdout.

## Files

| file | purpose |
|---|---|
| `protocol.py` | Pure. The frames Retell sends (`call_details`, `ping_pong`, `update_only` word by word, `response_required`), the speaking pace (150 words a minute), every timeout, the ask-set check, and `TurnWatch`, which times one reply on the client side. |
| `report.py` | Pure. Nearest-rank p50/p90 per measure, per kind and overall; the results line; a real call's join with Retell's `vendor_latency`. |
| `run.py` | The Auto Live CLI: the socket, the database. |
| `quick_ask_protocol.py` | Pure. The quick-ask's two request bodies, the turn reference, the phone's sentence rule, `QuickAskWatch` (times one reply's stream), the measures and the server's split read from a reply's trace. |
| `quick_ask.py` | The quick-ask CLI (see [Quick-ask](#quick-ask)): HTTP, and the database only when it runs beside one. |

## The ask set

A JSON list on stdin, 20 asks, 5 of each kind:

```json
[
  {"kind": "small_talk", "text": "hey Auto how's it going"},
  {"kind": "knowledge",  "text": "what does our refund policy say about digital goods"},
  {"kind": "board",      "text": "what's on my board today"},
  {"kind": "tool",       "text": "what's the weather in Dublin right now"}
]
```

`kind` is one of `small_talk`, `knowledge`, `board`, `tool`; `text` is what
is spoken. A bad row stops the run before anything is created, naming the
row's index (0-based).

## A simulator run

Inside the backend container, on the local stack, with Auto Live on
(Settings → Voice: `voice.live_enabled`; the runner refuses otherwise):

```bash
docker exec -i automatos_backend python -m scripts.eval.voice_latency.run \
  --workspace <workspace id> --user <your users.id> --sha "$(git rev-parse --short HEAD)" \
  < ~/.automatos-analyst/voice-eval/asks.json
```

What it does:

1. Creates a chat for the run (a "Voice call — …" thread in that workspace)
   and a `minted` `voice_calls` row with provider `sim` bound to it.
2. Opens `ws://localhost:8000/api/voice/retell/llm-websocket/<call_id>`
   (`--port` / `--host` override; 8000 is uvicorn's port inside the
   container) and answers the server's `config` frame with `call_details`,
   then pings every 2 s, as Retell does.
3. For each ask, in order: one `update_only` per word at 150 words a minute,
   then `response_required`. It times, from `response_required`, the first
   content frame (the one-time "One moment." counts), Auto's own first frame
   (it never does) and `content_complete`. Before the next ask it waits for
   the turn's `voice_turns` row, so the next ask never barges in.
4. Ends the sim call at 0 s (it never counts against the workspace's minutes),
   reads the call's `voice_turns` rows for `prep_ms` and `tools_before_speech`,
   and prints a JSON summary, then the results line.

`--sha` names the build in the results line (the container carries no git
checkout); without it the line says `unknown`.

Append the last printed line to `~/.automatos-analyst/voice-eval/results.md`.

## The results line

```
<ISO time> · <build SHA> · overall: first_frame p50/p90 · first_auto_frame p50/p90 · full p50/p90 · prep p50/p90 · silent N | small_talk: … | knowledge: … | board: … | tool: …
```

Whole milliseconds, nearest-rank percentiles. `-` means no turn had that
measure. `silent` counts turns where none of Auto's words arrived. The server
columns (`prep`) are joined to the asks by order, and only when the call has
exactly one row per ask; otherwise they are left empty and the JSON says
`"server_rows_matched": false`.

## A real call

Press Live on /chat and read the 20 asks in order. Then:

```bash
docker exec -i automatos_backend python -m scripts.eval.voice_latency.run --call <call_id> --sha "$(git rev-parse --short HEAD)"
```

No socket, no insert: it joins that call's `voice_turns` rows with its
`voice_calls.vendor_latency` (PRD-258 US-002) and prints the same line shape
for a single `call` group (full = the turn's `total_ms`; silent = no audio
delivered), followed by Retell's own `vendor e2e · asr · llm · tts` p50/p90.
`vendor_latency` arrives with Retell's `call_analyzed` event, a little after
the call ends.

## Quick-ask

Times the voice screen's quick-ask (Automatos Go) without a
person or a phone. Per ask it does what the phone does over HTTP, and it reads
the same ask set on stdin.

```bash
docker exec -i automatos_backend python -m scripts.eval.voice_latency.quick_ask \
  --workspace <workspace id> --sha "$(git rev-parse --short HEAD)" \
  < ~/.automatos-analyst/voice-eval/asks.json
```

| option | what it does |
|---|---|
| `--workspace` | The workspace the asks run in. Required. |
| `--agent <id>` | Sends `agentId`. A turn that names its agent takes another route (no classifier) and Go names none, so time both. |
| `--new-chat-each` | A new chat for every ask. Left out, the run keeps one chat, as the voice screen does. |
| `--no-speech` | Times the chat alone; no speech request is made. |
| `--unit-ends on\|off` | Forces the sentence rule. Left out, it follows the server's `spoken_unit_ends` capability, so a run cuts the first sentence the way a phone on that server would. |
| `--host`, `--port` | Where the backend is (`localhost:8000` inside the container). `--host` may be a whole `https://…` base URL; the port is then not used. |
| `--token-file <path>` | Hosted mode, below. |
| `--sha` | Names the build in the results line. |

What it does for each ask, in order:

1. `POST /api/chat` with `spokenMode: true` and a `turnRef`
   (`vt-<send time in ms, base 36>-<the ask's number, base 36>`). The chat id
   comes from the stream's chat-id frame and is sent from the second ask on.
2. Reads the stream and times, from the send: the response headers, the first
   text frame, the first complete sentence and the end of the stream. The
   sentence rule is the phone's (`. ! ? …` then whitespace; `Dr.`, `e.g.` and
   an initial do not end one). With `spoken_unit_ends`, a terminator at the end
   of the text so far ends a sentence too. A reply with no sentence end is one
   sentence when the stream ends.
3. `POST /api/voice/speech` for that first sentence with the same `turnRef`,
   in the default voice; times the first body byte and the last, from that
   request's send. It is sent after the chat stream has ended, not beside it,
   so the two never share the server.
4. Waits a second, then the next ask.

It refuses (exit 2) when the server's discovery lists no `spoken_mode`, or no
`tts` and `--no-speech` was not given.

### Measures

Whole milliseconds, nearest-rank p50/p90, per kind and overall; `-` where no
ask had the measure.

| measure | from | is |
|---|---|---|
| `headers` | chat send → response headers | auth, the save, routing |
| `first_text` | chat send → first text frame | on a spoken turn, the first whole speech unit |
| `first_sentence` | chat send → first complete sentence | when a phone could first ask for audio |
| `tts_first_byte` | speech send → first body byte | the speech route's wait plus synthesis of the start |
| `tts_total` | speech send → last body byte | the whole file |
| `full` | chat send → end of the stream | the whole reply |
| `go_first_audio` | `first_sentence + tts_total` | **today's phone**: it plays a sentence once its whole file is in |
| `streamed_first_audio` | `first_sentence + tts_first_byte` | what playing while downloading could reach |
| `held_first_audio` | `full + tts_total` | **a client that holds the whole reply** and checks it before any is spoken |

The three derived numbers are sums of parts measured one after the other. One
is empty whenever a part is (no sentence, a refused speech request,
`--no-speech`); it is never estimated. They leave out what only a phone has:
the network from the phone, the file write and the player's start.

Beside a database (no `--token-file`), the run then reads each reply's
`context_trace["timing"]` (by workspace and chat, then by `turnRef`) and adds
the server's own split, each name led by `srv_`: the marks `authorised`,
`routed`, `locked`, `agent`, `model_start`, `first_response`, `first_text`,
the stages `agent_context`, `tools`, `context`, `composio`, `knowledge`,
`model`, and the count `model_calls`. The stored trace is a snapshot taken
when the reply was saved, so a one-unit reply can lack `srv_first_text`: the
runner's own `first_text` is the measure, the server's is the split. With
`CHAT_TURN_TIMING_ENABLED` off there is no trace and every `srv_` measure is
`-`; the JSON's `server_traces` says how many asks had one.

### The results line

```
<ISO time> · <build SHA> · overall: headers p50/p90 · first_text p50/p90 · … · held_first_audio p50/p90 [· srv_… p50/p90] · silent N | small_talk: … | … · server_marks=read
```

The last field is `server_marks=read` or `server_marks=unavailable`. It is
`unavailable` in hosted mode, and also when the read after the run fails (the
runner has no database within reach, for example from the host against a
container): the run's own measures are still reported, and a
`quick_ask_server_marks_unreadable` warning names the error type. Neither
the JSON nor the line holds what was asked or answered.

### Before and after a change

1. On the build before: the 20 asks, once as above and once with `--agent`.
2. The same two runs on the build after: same server, same ask file.
3. Compare `go_first_audio`, `first_sentence` and the `srv_` split.

### Hosted mode

```bash
python -m scripts.eval.voice_latency.quick_ask --host https://<api host> \
  --workspace <workspace id> --token-file <path to a file holding a bearer token> \
  < ~/.automatos-analyst/voice-eval/asks.json
```

- The token file is read again before every chat and speech request, because
  a session token is short-lived and a 20-ask run is longer: use a paired
  device's token, or keep the file refreshed (write a new file and rename it
  over the old one, so it is never read half-written). The token is never an
  argument, a log line or output. Keep the file outside this repo.
- A token is sent only over https, or to this machine (`localhost`, `127.0.0.1`,
  `::1`). A run that would send it over plain http to another host is refused
  before any request.
- If the token file is empty or unreadable before an ask, the run stops there
  and still reports the asks already timed; the JSON's `asks_not_sent` says
  how many were left. If it is unreadable before the first ask, the run is
  refused (exit 2). If only a speech request finds it unreadable, that ask
  keeps its chat times and has no speech times.
- There is no database, so every server measure is empty and the line ends
  `server_marks=unavailable`. The server's `chat_turn_timing` log lines of the
  run hold the same numbers, by chat and `ref`.
- It still needs the orchestrator's Python environment (it imports this
  package).

### What a run leaves behind

Real model calls (credits), one chat with its messages (one per ask with
`--new-chat-each`), and whatever the turn's memory step keeps from those
turns. Run it in a workspace kept for this, outside demo hours.
