# Benchmarks

This directory holds **committed snapshots** of tool-routing eval runs.
Day-to-day scratch results (`../results/`) are gitignored; only blessed
benchmark runs land here.

## Layout

Each snapshot is a date-stamped subfolder:

```
benchmarks/
├── 2026-05-02-smoke-gpt41mini/
│   ├── results.jsonl          # raw per-cell results
│   ├── report.md              # rendered score.py output
│   ├── summary.csv            # per (model, mode) aggregates
│   ├── eval_set.jsonl         # exact query set used (snapshot)
│   ├── models.yaml            # model matrix snapshot (with prices at run time)
│   └── notes.md               # what this run was, why, any caveats
└── 2026-05-09-full-matrix/
    └── ...
```

The eval set + models.yaml are snapshotted alongside results so a
future run can be compared apples-to-apples even after queries are
added or model prices change.

## How to promote a run

After running `run_eval` + `score`, snapshot the output:

```bash
cd orchestrator
python -m scripts.eval.tool_routing.snapshot \
    --label "full-matrix" \
    --notes "First full sweep across all 9 models, both modes."
```

This copies `results/` + the current `eval_set.jsonl` and `models.yaml`
into `benchmarks/YYYY-MM-DD-<label>/` and writes a `notes.md`.

## Comparing across snapshots

Pass two snapshot dirs to a (future) `compare.py` to diff per-(model, mode)
metrics. For now, manual diff of the summary CSVs is fine.
