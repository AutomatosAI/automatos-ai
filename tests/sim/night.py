"""The overnight runner (PRD-247 S0.6). Report-only.

    python3 -m tests.sim.night run --pack smoke            one pack, now
    python3 -m tests.sim.night run --pack auto             tonight's pack by rotation
    python3 -m tests.sim.night packs                       list and validate packs
    python3 -m tests.sim.night latest                      where the last scorecard is

One workspace per run, the cheap model pinned on everything the run seeds,
a dollar ceiling checked before every scenario, the global LLM tiers put
back in ``finally``, the workspace purged unless ``--keep``. Exit 0 when the
pack completed, 3 when the budget stopped it, 2 when the harness itself
could not run — a bad scorecard is still exit 0: the report is the product.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .answerer import load_persona
from .api import Api, ApiError, Trace
from .config import ConfigError, Settings, check_docker_target, ensure_dirs, load_settings, public_settings
from .cost import CostError, models_seen, summarise, usage_rows, workspace_total
from .driver import run_scenario
from .judge import judge_available, judge_output
from .packs import Pack, PackError, list_packs, load_pack, rotation_for
from .results import RunContext, ScenarioResult, now_iso
from .score import findings, render_markdown, score_pack, score_scenario
from .store import (RunLocked, acquire_run_lock, latest_run_dir, new_run_dir, record_campaign, update_latest,
                    write_json, write_run)
from .workspace import (SimWorkspace, WorkspaceError, apply_llm_settings, provision, purge, restore_llm_settings,
                        seed_agents, set_agent_models, snapshot_llm_settings)

EXIT_OK, EXIT_HARNESS, EXIT_BUDGET = 0, 2, 3


class Log:
    def __init__(self, path: Path):
        self.path = path

    def __call__(self, message: str) -> None:
        line = f"{now_iso()} {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def _rows_in_window(rows: tuple[dict[str, Any], ...], start: str, end: str) -> tuple[dict[str, Any], ...]:
    return tuple(r for r in rows if start <= str(r.get("created_at") or "")[:19] <= end)


def _cost_rows(settings: Settings, ws: SimWorkspace, since: str, log: Log) -> tuple[tuple[dict[str, Any], ...], str | None]:
    try:
        return usage_rows(settings, ws.id, since), None
    except (CostError, ValueError) as exc:
        log(f"cost unavailable: {exc}")
        return (), str(exc)


def _score_one(settings: Settings, res: ScenarioResult, rows: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    window = _rows_in_window(rows, res.started_at[:19], res.ended_at[:19])
    cost = summarise(rows, res.execution_ids) if res.execution_ids else summarise(window)
    cost["window"] = summarise(window)
    verdict = None
    if res.kind != "crud" and judge_available(settings):
        verdict = judge_output(settings, brief=res.brief, expect=res.expect, output=res.evidence_text)
    return {**res.to_dict(), "cost": cost, "verdict": verdict, "scores": score_scenario(res, cost, verdict),
            "findings": list(findings(res, cost))}


def _run_pack(settings: Settings, pack: Pack, only: set[str], run: dict[str, Any], run_dir: Path, log: Log) -> None:
    trace: Trace = run["_trace"]
    ws: SimWorkspace = run["_ws"]
    api = Api(settings.api_url, settings.api_key or None, ws.id, trace)
    run["_api"] = api
    run["auth"] = "static X-Api-Key + X-Workspace-ID" if settings.api_key else "anonymous local operator, scoped by X-Workspace-ID"
    log(f"auth: {run['auth']}")
    if settings.set_global_models:
        snapshot = snapshot_llm_settings(api)
        write_json(run_dir / "llm-settings-snapshot.json", list(snapshot))
        # The snapshot is on the run BEFORE anything is applied: if apply fails half-way,
        # _finish still has what it needs to put the rows back.
        run["global_llm"] = {"snapshot": list(snapshot), "applied": [], "restored": False}
        run["global_llm"]["applied"] = list(apply_llm_settings(api, snapshot, settings.model_provider, settings.model_id))
        log(f"global LLM tiers: {len(run['global_llm']['applied'])} rows pointed at {settings.model_provider}/{settings.model_id} (snapshot saved)")
    agents = seed_agents(api, pack.agents)
    run["agents"] = {k: {"id": a.get("id"), "name": a.get("name")} for k, a in agents.items()}
    run["agent_models"] = list(set_agent_models(api, agents, settings.model_provider, settings.model_id))
    log(f"seeded {len(agents)} agents; model pinned on {sum(1 for m in run['agent_models'] if m['ok'])}")
    ctx = RunContext(settings=settings, api=api, trace=trace, workspace_id=ws.id, agents=agents,
                     persona=load_persona(pack.persona), run_started=run["started_at"])
    budget = min(settings.budget_usd, pack.budget_usd or settings.budget_usd)
    for sc in pack.scenarios:
        if only and sc.id not in only:
            continue
        rows, _ = _cost_rows(settings, ws, run["started_at"], log)
        spent = workspace_total(rows)
        if spent >= budget:
            run["status"] = "budget_stop"
            run["notes"].append(f"stopped before '{sc.id}': ${spent:.4f} spent of the ${budget:.2f} ceiling")
            log(run["notes"][-1])
            break
        log(f"▶ {sc.id} ({sc.kind}) — ${spent:.4f} of ${budget:.2f} spent so far")
        res = run_scenario(ctx, sc)
        rows, _ = _cost_rows(settings, ws, run["started_at"], log)
        scored = _score_one(settings, res, rows)
        run["scenarios"].append(scored)
        s = scored["scores"]
        log(f"■ {sc.id}: {res.outcome} in {res.duration_s}s · usability {s['usability']} · ${s['cost_usd']:.4f} · {s['calls']} calls · findings {len(scored['findings'])}")
        write_run(run_dir, _public(run))
    run["status"] = run.get("status") or "completed"


def _public(run: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in run.items() if not k.startswith("_")}


def _finish(settings: Settings, run: dict[str, Any], run_dir: Path, log: Log) -> None:
    api: Api | None = run.get("_api")
    ws: SimWorkspace | None = run.get("_ws")
    snapshot = (run.get("global_llm") or {}).get("snapshot") or []
    if api and snapshot and not run["global_llm"].get("restored"):
        try:
            restore_llm_settings(api, snapshot)
            run["global_llm"]["restored"] = True
            log("global LLM tiers restored")
        except ApiError as exc:
            run["notes"].append(f"COULD NOT RESTORE GLOBAL LLM SETTINGS: {exc}. Snapshot: {run_dir / 'llm-settings-snapshot.json'}")
            log(run["notes"][-1])
    rows: tuple[dict[str, Any], ...] = ()
    if ws:
        rows, note = _cost_rows(settings, ws, run["started_at"], log)
        run["cost"] = {"workspace_total_usd": workspace_total(rows), "rows": len(rows), "models_seen": list(models_seen(rows)),
                       "source": "llm_usage" if note is None else "unavailable", "note": note}
        drift = [m for m in models_seen(rows) if m != settings.model_id]
        if drift:
            run["findings_extra"] = [{"scenario": "*", "kind": "model_drift", "basis": "trace",
                                      "text": f"models other than the pinned {settings.model_id} spent in this workspace: {', '.join(drift)}",
                                      "evidence": {"models_seen": list(models_seen(rows))}}]
    run["scorecard"] = score_pack(run["scenarios"], (run.get("cost") or {}).get("workspace_total_usd", 0.0))
    run["findings"] = [f for sc in run["scenarios"] for f in sc.get("findings", [])] + run.pop("findings_extra", [])
    if ws:
        if settings.keep_workspace:
            run["workspace"]["purged"] = False
            run["notes"].append(f"workspace kept: purge with  python3 -m tests.sim.night purge --workspace-id {ws.id}")
        else:
            try:
                run["workspace"]["purge_result"] = purge(settings, ws.id)
                run["workspace"]["purged"] = True
                log(f"workspace {ws.slug} purged")
            except WorkspaceError as exc:
                run["workspace"]["purged"] = False
                run["notes"].append(f"PURGE FAILED for {ws.id}: {exc}")
                log(run["notes"][-1])
    run["ended_at"] = now_iso()
    public = _public(run)
    write_run(run_dir, public)
    (run_dir / "scorecard.md").write_text(render_markdown(public), encoding="utf-8")
    run["_trace"].dump_jsonl(run_dir / "trace.jsonl")
    record_campaign(public, run_dir)
    update_latest(run_dir)


def _new_run(run_id: str, pack: Pack, settings: Settings, note: str | None) -> dict[str, Any]:
    notes = [note] if note else []
    if not judge_available(settings):
        notes.append("no judge: OPENROUTER_API_KEY unset or SIM_JUDGE=0 — quality/usefulness fall back to cheap checks")
    return {
        "run_id": run_id, "pack": pack.name, "pack_description": pack.description, "started_at": now_iso(), "ended_at": None,
        "status": None, "settings": public_settings(settings), "model": {"provider": settings.model_provider, "model_id": settings.model_id},
        "workspace": {}, "agents": {}, "agent_models": [], "global_llm": {}, "scenarios": [], "cost": {}, "scorecard": {},
        "findings": [], "notes": notes, "_trace": Trace(),
    }


def cmd_run(args: argparse.Namespace) -> int:
    try:
        settings = load_settings({"budget_usd": args.budget, "keep_workspace": args.keep or None, "model_id": args.model,
                                  "model_provider": args.provider, "api_url": args.api_url,
                                  "judge": False if args.no_judge else None,
                                  "set_global_models": False if args.no_global_models else None})
        ensure_dirs()
        pack_name, note = (rotation_for(date.today()) if args.pack == "auto" else (args.pack, None))
        pack = load_pack(pack_name)
    except (ConfigError, PackError) as exc:
        print(f"cannot run: {exc}", file=sys.stderr)
        return EXIT_HARNESS
    run_id, run_dir = new_run_dir(pack.name)
    log = Log(run_dir / "run.log")
    log(f"automatos-sim {__version__} · pack {pack.name} · run {run_id} · model {settings.model_provider}/{settings.model_id}")
    run = _new_run(run_id, pack, settings, note)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")  # one workspace per run, never a purged one by slug
    try:
        lock = acquire_run_lock()
    except RunLocked as exc:
        log(f"cannot run: {exc}")
        return EXIT_HARNESS
    try:
        run["docker_host"] = check_docker_target(settings) or "local socket"
        ws = provision(settings, slug=f"sim-{pack.name}-{stamp}", name=f"SIM {pack.name} {stamp}", purpose=f"sim:{pack.name}")
        run["_ws"] = ws
        run["workspace"] = {"id": ws.id, "slug": ws.slug, "name": ws.name, "purged": None}
        log(f"workspace {ws.slug} = {ws.id}")
        _run_pack(settings, pack, set(args.only or ()), run, run_dir, log)
    except (WorkspaceError, ApiError, ConfigError) as exc:
        run["status"] = "aborted"
        run["notes"].append(f"aborted: {exc}")
        log(run["notes"][-1])
    except Exception as exc:  # noqa: BLE001 — the finally block must still restore and purge
        run["status"] = "aborted"
        run["notes"].append(f"aborted: {type(exc).__name__}: {exc}")
        log(traceback.format_exc())
    finally:
        _finish(settings, run, run_dir, log)
        lock.close()
    card = run["scorecard"]
    log(f"done: {run['status']} · {card.get('ok')}/{card.get('scenarios')} ok · usability {card.get('usability')} · "
        f"quality {card.get('quality')} · usefulness {card.get('usefulness')} · ${card.get('cost_usd', 0):.4f}")
    log(f"scorecard: {run_dir / 'scorecard.md'}")
    return {"completed": EXIT_OK, "budget_stop": EXIT_BUDGET}.get(run["status"], EXIT_HARNESS)


def cmd_packs(_: argparse.Namespace) -> int:
    failures = 0
    for name in list_packs():
        try:
            pack = load_pack(name)
            print(f"{name:20s} {len(pack.agents)} agents · {len(pack.scenarios)} scenarios · {pack.description}")
        except PackError as exc:
            failures += 1
            print(f"{name:20s} INVALID: {exc}")
    tonight, note = rotation_for(date.today())
    print(f"\ntonight by rotation: {tonight}" + (f" ({note})" if note else ""))
    return EXIT_HARNESS if failures else EXIT_OK


def cmd_latest(_: argparse.Namespace) -> int:
    run_dir = latest_run_dir()
    if not run_dir:
        print("no runs yet")
        return EXIT_HARNESS
    print(run_dir / "scorecard.md")
    return EXIT_OK


def cmd_purge(args: argparse.Namespace) -> int:
    try:
        settings = load_settings({})
        print(purge(settings, args.workspace_id))
    except (ConfigError, WorkspaceError) as exc:
        print(f"purge failed: {exc}", file=sys.stderr)
        return EXIT_HARNESS
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tests.sim.night", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run one pack now")
    run.add_argument("--pack", default="auto", help="pack name, a .toml path, or 'auto' for tonight's rotation")
    run.add_argument("--budget", type=float, help="dollar ceiling for the workspace (default SIM_BUDGET_USD)")
    run.add_argument("--keep", action="store_true", help="do not purge the workspace at the end")
    run.add_argument("--only", action="append", help="run only this scenario id (repeatable)")
    run.add_argument("--model", help="model id to pin (default SIM_MODEL_ID)")
    run.add_argument("--provider", help="provider for the pinned model (default SIM_MODEL_PROVIDER)")
    run.add_argument("--api-url", help="local API (default SIM_API_URL)")
    run.add_argument("--no-judge", action="store_true", help="skip the standalone judge even if a key is set")
    run.add_argument("--no-global-models", action="store_true", help="leave the global LLM tiers alone (Auto keeps its usual model)")
    run.set_defaults(fn=cmd_run)
    sub.add_parser("packs", help="list and validate packs").set_defaults(fn=cmd_packs)
    sub.add_parser("latest", help="path of the latest scorecard").set_defaults(fn=cmd_latest)
    purge_cmd = sub.add_parser("purge", help="purge a kept sim workspace")
    purge_cmd.add_argument("--workspace-id", required=True)
    purge_cmd.set_defaults(fn=cmd_purge)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
