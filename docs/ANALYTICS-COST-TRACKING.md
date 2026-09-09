# Analytics cost tracking — what `llm_usage` records and how the page reads it

*2026-09-09. Applies to both editions; written for the local edition, where one
operator runs API keys (paid), free trial routes (NVIDIA) and their own Claude
Code subscription side by side and needs to see what each did and cost.*

## One row per call

Every call to a model API, an embeddings API, a rerank API, and every Claude
Code session turn writes one row to `llm_usage` (`core/llm/usage_tracker.py`).
Nothing is exempt: a free call is recorded at $0, a subscription session at $0
with its tokens, an errored call with `status='error'` and zero tokens.

| Column | Meaning |
|---|---|
| `provider` | The **serving provider slug**: a registry API (`openrouter`, `nvidia`, `anthropic`, `openai`, `cohere`…) or a session runtime (`claude_code`, `codex`). Never a vendor name. |
| `model_id` | The model as the provider names it (`moonshotai/kimi-k3`, `claude-fable-5`). |
| `tier` | The provider's registry **kind** — `direct`, `aggregator`, `hosted_open` — or `subscription`. |
| `request_type` | The **lane** that spent: `chat`, `board_task`, `mission`, `heartbeat`, `scheduled_task`, `session`, `embedding`, `rerank`, `recipe`, `watch`, `planner`, `verifier`, `digest`, `memory_distill`… |
| `execution_id` | `<kind>:<id>` of the thing that spent — `mission:<run>`, `board_task:<ticket>`, `chat:<conversation>`, `session:<session_id>`. |
| `input_tokens` | The **full prompt** (fresh + cached + written) on every provider. |
| `cache_read_tokens` / `cache_write_tokens` | The prompt-cache breakdown of `input_tokens`. |
| `total_cost` | Priced per **route** (provider × model). A provider-reported figure (OpenRouter returns the credits charged) wins over any estimate. |
| `is_byok` | The call ran on the user's own key or plan (sessions are always `true`). |

**Billing** is derived, not stored: `metered` (a paid API), `free` (the
registry marks the provider free — NVIDIA's trial), `subscription` (a session
runtime). `describe_usage_provider()` in `core/llm/providers.py` is the one
place that turns a slug into its label, kind and billing.

## Attribution: the lane, task-locally

`core/llm/usage_context.py` carries `request_type` / `execution_id` /
`agent_id` on a `ContextVar`. `AgentFactory.execute_with_prompt` opens a scope
from its `context` (`source`, `task_id`, `mission_id`…), the chat turn opens
`chat:<id>`, and every `LLMManager` used inside — the agent's, a helper's, an
embedding — books to that scope. Two runs of the same agent at once never
overwrite each other (the old per-manager dict did).

## Pricing fallbacks (never a silent $0 on a paid route)

1. the route row in `llm_models` (`serving_provider`, `model_id`);
2. any `llm_models` row for the id × the registry's price multiplier;
3. the OpenRouter catalogue cache;
4. the static estimate map — only when a key actually matches the model;
5. else $0, logged at INFO with the route so the gap is visible.

A free provider's multiplier (0) and a subscription's `cost_override=(0, 0)`
short-circuit all of it. Cache reads/writes are re-priced at the vendor's
published multipliers (Anthropic 0.1× / 1.25×, OpenAI 0.5×).

## Sessions (Claude Code)

The host reads the transcript's per-model token totals (`transcript.py`) and
reports the **delta** for the turn: a snapshot at `SessionStart` for a resumed
ticket session, at launch for a Runtime Canvas terminal (`usage_before`), so a
`--resume` never re-books earlier turns. The backend books them in
`cli_host_service.book_session_usage` from `apply_result` (ticket runs,
lane `board_task`) and `TerminalClosed` (interactive, lane `session`).
Host contract 0.5.0.

## What the page must never do again

- sum the OpenRouter activity-sync rows (`request_type='activity_sync'`) with
  the per-call rows — they are OpenRouter's own daily report, kept for
  reconciliation and excluded from every aggregate (`_calls()` in
  `api/llm_analytics.py`);
- read the cumulative `Agent.model_usage_stats` blob for a period view — the
  Agents and LLM & Costs tabs use `/api/analytics/llm/usage?group_by=agent`
  for the selected period;
- merge the same vendor model served by two providers into one line — every
  chart, table and projection is keyed by route (`<model>@<provider>`).
