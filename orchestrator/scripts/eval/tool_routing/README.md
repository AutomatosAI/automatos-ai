# Tool-routing eval (PRD-138)

A multi-model harness for measuring whether **semantic action filtering**
(top-K) lets smaller models match frontier models on tool-routing accuracy
without the prompt bloat of dumping the full ~107-action catalog.

The hypothesis: with a focused, query-relevant catalog in the prompt,
**a 70B / Haiku-class model performs on par with GPT-5 / Opus** for tool
selection — at a fraction of the cost.

## Files

| file | purpose |
|---|---|
| `models.yaml` | Model matrix (frontier / mid / small) + embedding model + tunables |
| `eval_seed.yaml` | Hand-curated queries + their `correct_actions` |
| `seed_eval_set.py` | Validates seed queries against the live `ActionRegistry`, emits `eval_set.jsonl` |
| `prompt_builder.py` | Replicates production prompt, plus filtered top-K mode |
| `run_eval.py` | Cartesian runner over (model × mode × query), writes `results/results.jsonl` |
| `score.py` | Aggregations + markdown report (`results/report.md`) + CSV summary |
| `_registry_bootstrap.py` | Loads `ActionRegistry` without booting the full orchestrator package chain |
| `results/` | All eval output (gitignored) |
| `.embedding_cache.json` | Disk-cached embeddings, keyed by `model::sha256(text)` (gitignored) |

## Setup

You need:

1. The orchestrator venv (so `modules.tools.discovery.action_registry` is importable).
2. `OPENROUTER_API_KEY` — one key, all models, including the embedding model.
3. `pip install openai pyyaml` if not already present.

Run everything from the `orchestrator/` directory so PYTHONPATH resolves:

```bash
cd orchestrator
export OPENROUTER_API_KEY=sk-or-...
```

## Workflow

### 1. Seed the eval set

Validates `eval_seed.yaml` against the live registry, writes `eval_set.jsonl`:

```bash
python -m scripts.eval.tool_routing.seed_eval_set
```

If you add or rename an action in the registry, this will fail with the
specific query + action that no longer matches. Fix the seed YAML and re-run.

### 2. Run the eval

Default: every model in `models.yaml` × `full` and `filtered` modes × every query.

```bash
python -m scripts.eval.tool_routing.run_eval
```

Useful flags:

```bash
# Smoke test — one model, both modes, 5 cells
python -m scripts.eval.tool_routing.run_eval --models openai/gpt-4.1-mini --limit 5

# Filtered mode only (skip the expensive full-dump baseline)
python -m scripts.eval.tool_routing.run_eval --mode filtered

# Subset of models (comma-separated IDs from models.yaml)
python -m scripts.eval.tool_routing.run_eval --models anthropic/claude-haiku-4.5,openai/gpt-5-mini

# Plan-only — print what *would* run without calling any LLM
python -m scripts.eval.tool_routing.run_eval --dry-run
```

The runner is **resumable**: rows already in `results/results.jsonl` are
skipped. You can ctrl-C and re-run; only unfinished cells get filled in.
Embedding cache is flushed every 25 cells.

### 3. Score it

```bash
python -m scripts.eval.tool_routing.score
```

Writes:

* `results/report.md` — main per-(model, mode) table, Δ table, per-category table
* `results/summary.csv` — same data in CSV for plotting

### 4. Snapshot the run for benchmarking

`results/` is gitignored — it's scratch. To preserve a run for future
comparison, promote it to a date-stamped, committed benchmark snapshot:

```bash
python -m scripts.eval.tool_routing.snapshot \
    --label "full-matrix" \
    --notes "First full sweep, all 9 models, both modes."
```

This copies `results/`, `eval_set.jsonl`, and `models.yaml` into
`benchmarks/YYYY-MM-DD-<label>/` so the benchmark is reproducible
even after queries are added or model prices change. See
`benchmarks/README.md` for the full convention.

## What the metrics mean

* **accuracy** — top-1: did the LLM's chosen action appear in the ground-truth list?
* **in-set rate** — only meaningful for `filtered` mode: did the prompt at
  least *surface* a correct action? If `in_set` is high but `accuracy` is
  low, the filter is fine and the LLM is the bottleneck. If both are low,
  the embedding ranker missed the action entirely.
* **prompt tokens (mean)** — what the LLM read on each call. Full mode is
  the bloat baseline; filtered should be ~5-10x smaller.
* **$/correct** — the metric that matters for production decisions. A model
  with 90% accuracy at $0.001/correct beats one with 95% at $0.05/correct.
* **p95 latency** — small-model latency is sensitive to prompt size; expect
  filtered mode to win here for every model.

## Reading the Δ table

The pair-diff table shows `filtered − full` for each model:

* `Δ accuracy = +12pp` on a small model means semantic routing **closed
  the gap** with frontier — exactly the hypothesis.
* `Δ accuracy ≈ 0` on a frontier model means full-dump prompts already
  saturate that model's attention budget. Filtered still wins on cost
  and latency, just not accuracy.
* `Δ accuracy < 0` is the signal we'd care about: it would mean the
  filter is dropping correct actions before the LLM ever sees them.
  That's the case where we'd need to bump `top_k` or improve embedding
  text composition.

## Adding queries

Edit `eval_seed.yaml`. Each entry needs:

```yaml
- q: "natural language query"
  correct_actions: [platform_action_name, ...]   # at least one
  category: agents | analytics | ...              # for slicing
  difficulty: easy | ambiguous | paraphrase | cross
  notes: "optional — surface what's tricky about this query"
```

Then re-run `seed_eval_set.py`. The validator catches typos against the
live registry. You can keep growing the set; the runner picks up new
queries automatically and only runs cells not already in `results.jsonl`.

## Knobs

In `models.yaml`:

* `top_k` (default 15) — how many actions filtered mode shows
* `temperature` (default 0.0) — 0 for reproducibility
* `max_tokens` (default 2048) — reasoning models (GPT-5, GPT-5-mini) burn tokens on chain-of-thought before the tool call; 256 starves them. Non-reasoning models stop early on their own.
* `request_timeout` (default 60s)

In `prompt_builder.py`:

* `_action_text_for_embedding()` controls what gets embedded per action.
  Currently: `name + description + tags + examples`. If actions get
  better tags/examples in the registry, embeddings improve automatically.

## What this *doesn't* measure

* **Multi-step routing** — every cell is a single tool call. Real agent
  loops chain actions; this eval is the per-step substrate, not the loop.
* **Parameter quality** — we score whether the *action name* is right,
  not whether the LLM populated `params` correctly. That's a separate
  eval and a different failure mode.
* **Tool execution** — the harness never actually executes any tool.
  No DB writes, no Composio calls, no workspace mutation. Pure routing.
* **Prompt-injection robustness** — adversarial queries are out of scope
  for this measurement.

## Cost expectations

A full sweep at the seed size (~45 queries × 9 models × 2 modes = 810 cells)
costs roughly **$2–$5 USD** at the listed model prices, dominated by
Opus 4.7 in full-dump mode. Filtered mode is ~10x cheaper across the board.
Embedding cost is negligible (~$0.01 for the whole run, embeddings
are cached after the first sweep).
