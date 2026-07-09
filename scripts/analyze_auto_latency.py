#!/usr/bin/env python3
"""Aggregate Auto chat-latency stats from Railway logs — the benchmark harness
behind reports/AUTO_LATENCY_BENCHMARK_2026-07-09.md.

Usage:
    railway logs --since 60m -n 5000 > /tmp/auto_bench.log
    python3 scripts/analyze_auto_latency.py /tmp/auto_bench.log [more.log ...]

Reads (all INFO-level, present in prod):
    [perf] rank_actions: ...            — narrowing embed timings + cache/timeout
    [perf] _rank_actions_for_dispatcher — narrowing totals
    [perf] _run_coroutine_blocking      — thread-bridge engagements (want: zero)
    [tool-trace X] Loaded N tools (...) — tool-load wall time
    [tool-trace X] execute_tool start/done — per-tool execution durations
    dispatcher enum narrowed/NOT narrowed  — narrowing outcomes
    LLM_CALL service=... model=...      — per-model LLM latency/tokens
    [req=...] spans                     — approximate per-request wall time

Sections print n/min/p50/p95/max; a missing log family just prints n=0, so
the script survives future log pruning. Deduplicates identical lines across
input files (overlapping captures are fine).
"""
import re
import sys
from collections import defaultdict
from datetime import datetime

TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3})")
REQ_RE = re.compile(r"\[req=([0-9a-f]+) ")
RANK_RE = re.compile(
    r"\[perf\] rank_actions: ensure_indexed=(\d+)ms query_embed=(\d+)ms "
    r"(?:cosine=(\d+)ms )?(?:\(TIMED OUT\) )?n_candidates=(\d+)(?: cache_hit=(\d))?")
TIMEOUT_RE = re.compile(r"query_embed=(\d+)ms \(TIMED OUT\)")
DISP_RE = re.compile(r"_rank_actions_for_dispatcher: get_index=(\d+)ms rank_actions(?:\(\+bridge\))?=(\d+)ms")
LOADED_RE = re.compile(r"Loaded (\d+) tools \(agent_id=([^,]*), denied=(\d+), candidates=(\d+), (\d+)ms\)")
BRIDGE_RE = re.compile(r"_run_coroutine_blocking (THREADED|DIRECT) (\d+)ms")
EXEC_START_RE = re.compile(r"\[tool-trace ([0-9a-f]+)\] execute_tool start tool=(\S+)")
EXEC_DONE_RE = re.compile(r"\[tool-trace ([0-9a-f]+)\] execute_tool done tool=(\S+) success=(\w+)")
NARROWED_RE = re.compile(r"dispatcher enum narrowed to (\d+) actions")
NOTNARROW_RE = re.compile(r"dispatcher enum NOT narrowed: reason=([^;]+);")
LLM_RE = re.compile(
    r"LLM_CALL service=(\S+) provider=\S+ model=(\S+) input_tokens=(\d+) "
    r"output_tokens=(\d+).*latency_ms=(\d+) status=(\S+)")


def pct(vals, p):
    if not vals:
        return None
    vals = sorted(vals)
    k = min(len(vals) - 1, max(0, int(round(p / 100 * (len(vals) - 1)))))
    return vals[k]


def fmt_stats(vals):
    if not vals:
        return "n=0"
    return (f"n={len(vals)} min={min(vals):.0f} p50={pct(vals, 50):.0f} "
            f"p95={pct(vals, 95):.0f} max={max(vals):.0f} (ms)")


def main(paths):
    lines, seen = [], set()
    for f in paths:
        for ln in open(f, encoding="utf-8", errors="replace"):
            ln = ln.rstrip("\n")
            if ln in seen:
                continue
            seen.add(ln)
            m = TS_RE.match(ln)
            if not m:
                continue
            lines.append((datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"), ln))
    lines.sort(key=lambda x: x[0])
    if not lines:
        print("no parseable log lines")
        return
    print(f"window {lines[0][0]} → {lines[-1][0]}  ({len(lines)} unique lines)\n")

    embeds, hits, timeouts, ensure = [], 0, [], []
    for _, l in lines:
        m = RANK_RE.search(l)
        if not m:
            continue
        ensure.append(int(m.group(1)))
        to = TIMEOUT_RE.search(l)
        if to:
            timeouts.append(int(to.group(1)))
        else:
            embeds.append(int(m.group(2)))
            if m.group(5) == "1":
                hits += 1
    print(f"narrowing query_embed (completed): {fmt_stats(embeds)}  cache_hits={hits}")
    print(f"narrowing query_embed TIMED OUT:   n={len(timeouts)} "
          + (f"bounded at {fmt_stats(timeouts)}" if timeouts else ""))
    print(f"ensure_indexed:                    {fmt_stats(ensure)}")
    print(f"dispatcher rank total:             "
          f"{fmt_stats([int(m.group(2)) for _, l in lines if (m := DISP_RE.search(l))])}")

    bridges = defaultdict(list)
    for _, l in lines:
        m = BRIDGE_RE.search(l)
        if m:
            bridges[m.group(1)].append(int(m.group(2)))
    if bridges:
        for kind, vals in sorted(bridges.items()):
            print(f"thread-bridge {kind}:             {fmt_stats(vals)}")
    else:
        print("thread-bridge:                     NONE (0 lines) ✓")

    print(f"tool-load wall ('Loaded N tools'): "
          f"{fmt_stats([int(m.group(5)) for _, l in lines if (m := LOADED_RE.search(l))])}")

    narrowed = [int(m.group(1)) for _, l in lines if (m := NARROWED_RE.search(l))]
    print(f"narrowed: n={len(narrowed)} enum_sizes={sorted(set(narrowed))}")
    reasons = defaultdict(int)
    for _, l in lines:
        m = NOTNARROW_RE.search(l)
        if m:
            reasons[m.group(1).strip()] += 1
    for reason, cnt in sorted(reasons.items()):
        print(f"NOT narrowed [{cnt}x]: {reason}")

    starts, durs, fails = {}, defaultdict(list), defaultdict(int)
    for t, l in lines:
        m = EXEC_START_RE.search(l)
        if m:
            starts[m.group(1)] = (m.group(2), t)
            continue
        m = EXEC_DONE_RE.search(l)
        if m and m.group(1) in starts:
            name, t0 = starts.pop(m.group(1))
            durs[name].append((t - t0).total_seconds() * 1000)
            if m.group(3) != "True":
                fails[name] += 1
    print("\nper-tool execute durations:")
    for name in sorted(durs, key=lambda n: -pct(durs[n], 50)):
        extra = f"  FAILS={fails[name]}" if fails.get(name) else ""
        print(f"  {name:36s} {fmt_stats(durs[name])}{extra}")

    llm = defaultdict(lambda: {"lat": [], "in": [], "out": [], "fail": 0})
    for _, l in lines:
        m = LLM_RE.search(l)
        if not m:
            continue
        a = llm[(m.group(1), m.group(2))]
        a["lat"].append(int(m.group(5)))
        a["in"].append(int(m.group(3)))
        a["out"].append(int(m.group(4)))
        if m.group(6) != "success":
            a["fail"] += 1
    print("\nLLM calls (by service/model, sorted by total latency):")
    for (svc, model), a in sorted(llm.items(), key=lambda kv: -sum(kv[1]["lat"])):
        print(f"  {svc:22s} {model:32s} n={len(a['lat']):3d} "
              f"lat_p50={pct(a['lat'], 50):6.0f}ms lat_max={max(a['lat']):6.0f}ms "
              f"in_p50={pct(a['in'], 50):6.0f} in_max={max(a['in']):6.0f} fails={a['fail']}")

    reqs = defaultdict(list)
    for t, l in lines:
        m = REQ_RE.search(l)
        if m:
            reqs[m.group(1)].append(t)
    spans = sorted(((max(v) - min(v)).total_seconds(), k) for k, v in reqs.items() if len(v) > 3)
    spans = [s for s in spans if s[0] > 0.5]
    if spans:
        vals = [s[0] for s in spans]
        print(f"\nper-request log spans (>0.5s, approx wall): n={len(spans)} "
              f"min={min(vals):.1f}s p50={pct(vals, 50):.1f}s max={max(vals):.1f}s")
        for s, k in spans[-8:]:
            print(f"  req={k} span={s:.1f}s")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1:])
