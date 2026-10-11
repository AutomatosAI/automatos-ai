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
| `run.py` | The CLI, the only part that does I/O. |

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
