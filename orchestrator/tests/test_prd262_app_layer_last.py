"""PRD-262 US-016: the API's app layer is written once, last.

The production stage used to copy the code in with `COPY . .` and then run
`chown -R automatos:automatos /app`, which stored all of /app a second time; the
`pip uninstall` RUN that US-014 removed sat between them. A code-only update
downloaded 14.7 MB in three layers over 1 MB. Now the stage creates the user and its
writable directories first, compiles the stdlib, copies the entrypoint and the
healthcheck, and ends with the code as its last filesystem layer, owned by the user.
Since PRD-262 US-032 that layer is `COPY --from=compile --chown=1000:1000 /app /app`:
the compile stage copies the context and compiles the boot modules, so code and
bytecode arrive together, in one layer (test_prd262_app_bytecode.py checks that stage).

This suite checks:

- the production stage's order: the user and its directories first, no `chown -R`,
  the code copied in last under the user's numeric id, then metadata only;
- every named volume a compose file mounts under /app lands on a directory the user
  owns, so a fresh volume starts writable;
- what a code change rebuilds, modelled on the committed Dockerfile with the layer
  sizes recorded from `edge` (tests/fixtures/prd262_edge_images.json): one layer, whose
  source share is within the update bar, where the old order cost 14.65 MB in three
  layers. The bytecode's share is measured, not modelled (US-032's experiment and
  images.yml's update bar on the real image);
- the update bar in .github/image-budgets.json, its verdict, and the `budget` command
  images.yml runs on every build, in the file's mode (warn until US-004);
- the stdlib compile's layer counts as the runtime (`system`), so a rebuilt compile is
  never filed as a code-only update.

Pure file reads: no Docker, no network.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = (ROOT / "orchestrator" / "Dockerfile").read_text(encoding="utf-8")
BUDGETS_PATH = ROOT / ".github" / "image-budgets.json"
BUDGETS = json.loads(BUDGETS_PATH.read_text(encoding="utf-8"))
FAIL_MODE = {**BUDGETS, "mode": "fail"}
WARN_MODE = {**BUDGETS, "mode": "warn"}  # report only, as until US-004
EDGE_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "prd262_edge_images.json"
EDGE = json.loads(EDGE_FIXTURE.read_text(encoding="utf-8"))
COMPOSE_FILES = ("docker-compose.yml", "docker-compose.dev.yml", "docker-compose.images.yml")

# The code and its boot modules' bytecode, from the compile stage (PRD-262 US-032).
APP_COPY = "COPY --from=compile --chown=1000:1000 /app /app"
# How BuildKit records that COPY in the image's history: no flags.
APP_COPY_HISTORY = "COPY /app /app"
WRITABLE = ("/app/logs", "/app/vector_stores", "/app/projects", "/app/exports", "/app/data")
# Instructions that only set image metadata, and those that can add a filesystem layer.
METADATA = {"USER", "EXPOSE", "HEALTHCHECK", "ENTRYPOINT", "CMD", "ENV", "LABEL", "STOPSIGNAL", "ARG", "SHELL"}
LAYER_MAKERS = {"RUN", "COPY", "ADD", "WORKDIR"}
GZIP = "application/vnd.oci.image.layer.v1.tar+gzip"
# The recorded `edge` predates US-014's stdlib compile, so its size is assumed: any
# layer over 1 MB shows what rebuilding it would cost. The user RUN writes /etc files
# and empty directories.
STDLIB_PYC_BYTES = 8_000_000
SMALL_LAYER_BYTES = 4_096
OLD_CODE_UPDATE_BYTES = 14_654_499  # COPY . ., pip uninstall, chown -R and four small layers
_CONTINUATION = re.compile(r"\\[ \t]*\n")
_CHOWN_R = re.compile(r"\bchown\s+(?:-\S*R|--recursive)")
_NAMED_VOLUME = re.compile(r"[A-Za-z0-9][\w.-]*")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "ci" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # before exec, as the scripts register their siblings
    spec.loader.exec_module(module)
    return module


layers = _load("image_layers")
report = _load("image_report")


def _instructions(text: str) -> list[str]:
    """The Dockerfile's instructions: continuation lines joined, comments dropped, spaces collapsed."""
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    return [" ".join(line.split()) for line in _CONTINUATION.sub(" ", code).splitlines() if line.strip()]


def _stage(name: str) -> list[str]:
    """A stage's instructions after its FROM."""
    instructions = _instructions(DOCKERFILE)
    start = next(i for i, line in enumerate(instructions) if re.fullmatch(rf"FROM \S+ as {name}", line, re.I))
    end = next((i for i in range(start + 1, len(instructions)) if instructions[i].upper().startswith("FROM ")),
               len(instructions))
    return instructions[start + 1:end]


PRODUCTION = _stage("production")
COMPILE = _stage("compile")


def _word(line: str) -> str:
    return line.split()[0].upper()


def _reads_the_whole_context(line: str) -> bool:
    """A COPY or ADD of the build context's root, which every code change invalidates."""
    return bool(re.fullmatch(r"(?:COPY|ADD) (?:--(?!from=)\S+ )*\. \S+", line))


def _carries_the_code(line: str) -> bool:
    """A layer every code change rebuilds: the context's root, or /app from the stage that copied it."""
    return _reads_the_whole_context(line) or line == APP_COPY


def _commands(run: str) -> list[list[str]]:
    """The words of each `&&`-joined command of a RUN."""
    return [command.split() for command in run.removeprefix("RUN ").split("&&")]


# --------------------------------------------------------------------------- the Dockerfile


def test_no_chown_r_is_left_in_the_production_stage():
    assert not [line for line in PRODUCTION if _CHOWN_R.search(line)]


def test_the_user_and_its_writable_directories_come_first():
    commands = _commands(PRODUCTION[0])
    assert commands[0] == ["useradd", "-m", "-u", "1000", "automatos"], PRODUCTION[0]
    mkdir = next(command for command in commands if command[0] == "mkdir")
    chown = next(command for command in commands if command[0] == "chown")
    assert mkdir[1] == "-p" and set(WRITABLE) <= set(mkdir[2:])
    assert chown[1] == "automatos:automatos", "the directories themselves: no -R"
    assert {"/app", *WRITABLE} <= set(chown[2:])
    assert commands.index(mkdir) < commands.index(chown)


def test_the_code_is_copied_in_under_the_users_numeric_id():
    uid = _commands(PRODUCTION[0])[0][3]
    assert APP_COPY == f"COPY --from=compile --chown={uid}:{uid} /app /app"
    assert [line for line in PRODUCTION if _carries_the_code(line)] == [APP_COPY]
    assert not [line for line in PRODUCTION if _reads_the_whole_context(line)], "the code comes from the compile stage"
    assert [line for line in COMPILE if _reads_the_whole_context(line)] == ["COPY . ."]


def test_the_code_is_the_last_filesystem_layer_and_only_metadata_follows():
    at = PRODUCTION.index(APP_COPY)
    makers = [i for i, line in enumerate(PRODUCTION) if _word(line) in LAYER_MAKERS]
    assert makers[-1] == at, "a code change must rebuild no layer after the code"
    after = PRODUCTION[at + 1:]
    assert after[0] == "USER automatos", after
    assert all(_word(line) in METADATA for line in after), after
    assert [line for line in PRODUCTION if _word(line) == "USER"] == ["USER automatos"]


def test_the_compile_and_the_two_scripts_sit_between_the_user_and_the_code():
    middle = PRODUCTION[1:PRODUCTION.index(APP_COPY)]
    copies = [line.split()[-1] for line in middle if _word(line) == "COPY"]
    assert copies == ["/usr/local/bin/docker-entrypoint.sh", "/usr/local/bin/docker-healthcheck.sh"]
    assert sum("compileall" in line for line in middle) == 1


class _ComposeLoader(yaml.SafeLoader):
    """Reads compose's !reset and !override merge tags as their plain value."""


def _plain(loader, suffix, node):
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    return loader.construct_scalar(node)


_ComposeLoader.add_multi_constructor("!", _plain)


def _mount(entry) -> tuple[str, str]:
    if isinstance(entry, dict):
        return (entry.get("source", "") if entry.get("type") == "volume" else ""), entry.get("target", "")
    source, target = (entry.split(":") + [""])[:2]
    return source, target


def _named_volumes_under_app() -> set[str]:
    """Where any compose file mounts a named volume under /app in the backend."""
    targets = set()
    for name in COMPOSE_FILES:
        services = yaml.load((ROOT / name).read_text(encoding="utf-8"), Loader=_ComposeLoader)["services"]
        for entry in (services.get("backend") or {}).get("volumes") or []:
            source, target = _mount(entry)
            if _NAMED_VOLUME.fullmatch(source) and target.startswith("/app/"):
                targets.add(target.rstrip("/"))
    return targets


def test_every_named_volume_under_app_lands_on_a_directory_the_user_owns():
    """Docker gives a fresh named volume its mount point's owner, so these start writable."""
    targets = _named_volumes_under_app()
    assert {"/app/logs", "/app/data"} <= targets, targets
    chown = next(command for command in _commands(PRODUCTION[0]) if command[0] == "chown")
    assert targets <= set(chown[2:])


def test_ci_checks_the_owners_inside_the_built_image():
    """test.yml's prod-image-parity builds the production stage and looks inside it as its user."""
    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "test.yml").read_text(encoding="utf-8"))
    steps = workflow["jobs"]["prod-image-parity"]["steps"]
    built = next(i for i, step in enumerate(steps) if "-t parity-be orchestrator/" in step.get("run", ""))
    check = next(i for i, step in enumerate(steps) if "find /app" in step.get("run", ""))
    assert built < check
    run = steps[check]["run"]
    assert run.startswith("docker run --rm --entrypoint sh parity-be -c '")
    assert '"$(id -u):$(id -g)" != "1000:1000"' in run
    assert "find /app \\( ! -user 1000 -o ! -group 1000 \\) -print" in run
    assert f"for dir in /app {' '.join(WRITABLE)}; do" in run
    assert '[ ! -w "$dir" ]' in run


# --------------------------------------------------------------------------- what a code change costs

EDGE_ROWS = layers.layer_rows(EDGE["api"]["manifest"], EDGE["api"]["config"])
RECORDED = {row["instruction"]: row["size"] for row in EDGE_ROWS}


def _digest(tag: str, position: int) -> str:
    return "sha256:" + hashlib.sha256(f"{tag}-{position}".encode()).hexdigest()


def _size(line: str) -> int:
    """The size `edge` recorded for this production-stage instruction, or the assumed one.

    The app layer gets the source's recorded size: the bytecode's share is measured on
    the real image (PRD-262 US-032), not assumed here.
    """
    if _carries_the_code(line):
        return RECORDED["COPY . ."]
    if "compileall" in line:
        return STDLIB_PYC_BYTES
    return RECORDED.get(line, SMALL_LAYER_BYTES)


def _committed_image() -> tuple[dict, dict]:
    """The API image as the committed Dockerfile builds it: `edge`'s base layers, then the production stage's."""
    base_end = next(i for i, row in enumerate(EDGE_ROWS) if row["instruction"] == "COPY /install /usr/local")
    rows = [(row["instruction"], row["size"]) for row in EDGE_ROWS[:base_end + 1]]
    rows += [(line, _size(line)) for line in PRODUCTION if _word(line) in LAYER_MAKERS]
    manifest = {"layers": [{"mediaType": GZIP, "digest": _digest("edge", i), "size": size}
                           for i, (_, size) in enumerate(rows)]}
    config = {"history": [{"created_by": f"{text} # buildkit"} for text, _ in rows]}
    return manifest, config


def _code_change(manifest: dict, config: dict) -> dict:
    """``manifest`` after a code-only change: every layer from the first that carries the code is rebuilt."""
    texts = [layers.instruction(entry) for entry in config["history"] if not entry.get("empty_layer")]
    first = next(i for i, text in enumerate(texts) if _carries_the_code(text))
    changed = copy.deepcopy(manifest)
    for position in range(first, len(changed["layers"])):
        changed["layers"][position]["digest"] = _digest("code", position)
    return changed


def _measured(manifest: dict, config: dict, edge: dict, name: str = "api") -> dict:
    """The parts of a measured report the update bar reads."""
    rows = layers.layer_rows(manifest, config)
    delta = layers.update_delta(rows, (layer["digest"] for layer in edge["layers"]))
    return {"name": name, "arch": "amd64", "bytes": 1, "layers": rows, "delta": delta}


def _old_code_update() -> dict:
    manifest, config = EDGE["api"]["manifest"], EDGE["api"]["config"]
    return _measured(_code_change(manifest, config), config, manifest)


def test_a_code_change_now_downloads_the_app_layer_alone_within_the_bar():
    """One new layer, the code and its bytecode. Its source share (5.3 MB in `edge`) is within
    the 6 MB bar; what the bytecode adds is measured on the real image (PRD-262 US-032, §9-Q7)."""
    edge, config = _committed_image()
    measured = _measured(_code_change(edge, config), config, edge)
    delta = measured["delta"]
    assert (delta["change"], delta["layers"], delta["bytes"]) == ("code-only", 1, RECORDED["COPY . ."])
    (fresh,) = delta["new_digests"]
    assert next(row for row in measured["layers"] if row["digest"] == fresh)["instruction"] == APP_COPY
    verdict = layers.update_verdict(measured, FAIL_MODE)
    assert verdict.level == "ok", verdict.message
    assert "5.3 MB, with 1 new layers over 1 MB, within its bar" in verdict.message


def test_the_old_order_cost_three_big_layers_and_misses_the_bar():
    measured = _old_code_update()
    assert measured["delta"]["change"] == "code-only"
    assert measured["delta"]["bytes"] == OLD_CODE_UPDATE_BYTES
    verdict = layers.update_verdict(measured, FAIL_MODE)
    assert verdict.level == "error"
    assert "downloads 14.7 MB, with 3 new layers over 1 MB" in verdict.message
    assert "8.7 MB over its 6.0 MB bar" in verdict.message
    assert "exactly 1 new layers over 1 MB" in verdict.message
    assert layers.update_verdict(measured, WARN_MODE).level == "warning"


# --------------------------------------------------------------------------- the update bar


def test_the_committed_bar_is_a_one_layer_six_mb_code_only_api_update():
    bar = BUDGETS["updates"]["api"]["code-only"]
    assert (bar["max_mb"], bar["layers_over_1_mb"]) == (6.0, 1)
    assert "US-016" in bar["reason"]


def test_every_update_bar_names_a_budgeted_image_and_a_kind_the_delta_files():
    kinds = {layers.change_kind({kind: 1}) for kind in (*layers.KINDS, layers.UNKNOWN)}
    for name, bars in BUDGETS["updates"].items():
        assert name in BUDGETS["budgets"], name
        for kind, bar in bars.items():
            assert kind in kinds, kind
            assert set(bar) <= {"max_mb", "layers_over_1_mb", "reason"}, bar
            assert bar["max_mb"] > 0 and bar["reason"].strip()


def test_a_build_with_no_bar_or_no_edge_passes_with_a_note():
    measured = _old_code_update()
    assert "no update delta" in layers.update_verdict({**measured, "delta": None}, FAIL_MODE).message
    frontend = layers.update_verdict({**measured, "name": "frontend"}, FAIL_MODE)
    assert frontend.level == "ok" and "no update bar for a code-only change" in frontend.message
    dependencies = {**measured["delta"], "change": "dependencies"}
    assert layers.update_verdict({**measured, "delta": dependencies}, FAIL_MODE).level == "ok"


def test_one_layer_over_the_size_bar_misses_it_on_size_alone():
    measured = _old_code_update()
    fresh = set(measured["delta"]["new_digests"])
    big = next(row for row in measured["layers"] if row["size"] > layers.MB and row["digest"] in fresh)
    one = {**measured["delta"], "bytes": 7_000_000, "new_digests": [big["digest"]]}
    verdict = layers.update_verdict({**measured, "delta": one}, FAIL_MODE)
    assert verdict.level == "error"
    assert verdict.message.endswith(": 1.0 MB over its 6.0 MB bar"), verdict.message


def test_the_budget_command_judges_the_update_bar_in_the_files_mode(tmp_path, capsys):
    path, summary = tmp_path / "api-amd64.json", tmp_path / "summary.md"
    path.write_text(json.dumps(_old_code_update()))
    warning, failing = tmp_path / "warn.json", tmp_path / "fail.json"
    warning.write_text(json.dumps(WARN_MODE))
    failing.write_text(json.dumps(FAIL_MODE))
    assert report.main(["budget", "--report", str(path), "--budgets", str(warning), "--summary", str(summary)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("::warning title=Update budget::this code-only update of api (amd64) downloads 14.7 MB")
    assert "Image size budget" not in out
    assert "**Update bar** (warn mode): this code-only update" in summary.read_text()
    assert report.main(["budget", "--report", str(path), "--budgets", str(failing)]) == 1
    assert capsys.readouterr().out.startswith("::error title=Update budget::")


def test_every_build_runs_the_budget_command_after_measuring():
    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "images.yml").read_text(encoding="utf-8"))
    steps = workflow["jobs"]["build"]["steps"]
    measure = next(i for i, step in enumerate(steps) if "image_report.py measure" in step.get("run", ""))
    check = next(i for i, step in enumerate(steps) if "image_report.py budget" in step.get("run", ""))
    assert measure < check
    assert "if" not in steps[check], "every image and arch is judged"
    assert "--budgets .github/image-budgets.json" in steps[check]["run"]


# --------------------------------------------------------------------------- layer kinds


def test_the_stdlib_compile_is_the_runtime_and_an_app_compile_is_code():
    compile_run = next(line for line in PRODUCTION if "compileall" in line)
    entry = {"created_by": f"RUN /bin/sh -c {compile_run.removeprefix('RUN ')} # buildkit"}
    assert layers.kind_of(layers.instruction(entry)) == "system"
    assert layers.kind_of("RUN python -m compileall -q /app") == "code"
    assert layers.kind_of("COPY --from=compile --chown=1000:1000 /app /app") == "code"
    assert layers.kind_of(APP_COPY) == "code"
    assert layers.kind_of(APP_COPY_HISTORY) == "code", "the app layer as the image's history names it"
