"""PRD-262 US-053 (the PR that merges before W8's): the media assets image and its job.

media-render's build is to copy the Kokoro files and the music library from
ghcr.io/<owner>/automatos-media-assets instead of fetching them from GitHub releases
and ende.app on every build. images.yml's ``media-assets`` job publishes the image;
``scripts/ci/media_assets.py`` fetches and checks the files, lays the image out, and
checks the published image again. This suite pins:

* the lists: until Gerard picks the D4 model (US-050) every candidate model is in the
  image, with the voices file and exactly the music library's tracks; every file
  ``kokoro/SHA256SUMS`` pins is there with the same sha256;
* the image: ``Apache-2.0 AND CC-BY-4.0``; a NOTICE with the Kokoro notice and every
  attribution line; the Apache terms without our copyright line; FROM scratch with one
  COPY per file, which builds exactly the tree ``verify`` accepts;
* the tool: a file whose bytes or size differ fails and is not kept; a transient error
  is retried, an oversized response is not; a published tree with one file different,
  missing or extra fails ``verify``;
* the job: only a manual run on main with ``media_assets`` runs it; that run builds no
  other image, has its own concurrency group and is skipped by smoke-images; it checks,
  publishes, checks again and only then signs; its labels carry the stage step's
  licences.

File reads and a fake opener: no network, no Docker.
"""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import re
import shutil
import sys
import urllib.error
from pathlib import Path, PurePosixPath

import pytest
import yaml

_REPO = Path(__file__).resolve().parents[2]
_WORKFLOWS = _REPO / ".github" / "workflows"
RELEASES = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
SOURCE = "https://example.test/download/"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _REPO / "scripts" / "ci" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # registered before exec, for its dataclass
    spec.loader.exec_module(module)
    return module


tool = _load("media_assets")
IMAGES = yaml.safe_load((_WORKFLOWS / "images.yml").read_text(encoding="utf-8"))
SMOKE = yaml.safe_load((_WORKFLOWS / "smoke-images.yml").read_text(encoding="utf-8"))
TRIGGERS = IMAGES.get("on", IMAGES.get(True))  # PyYAML reads the `on` key as True
JOBS = IMAGES["jobs"]
JOB = JOBS["media-assets"]
KOKORO, TRACKS = tool.load(tool.KOKORO_LIST, tool.MUSIC_MANIFEST)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _flat(text: str) -> str:
    return " ".join(str(text).split())


def _step(predicate) -> dict:
    return next(step for step in JOB["steps"] if predicate(step))


# --------------------------------------------------------------------------- the lists


def test_until_d4_is_picked_every_candidate_model_is_in_the_image():
    assert sorted(entry["url"] for entry in KOKORO["files"]) == sorted(RELEASES + path for path in (
        "model-files-v1.0/voices-v1.0.bin",
        "model-files-v1.0/kokoro-v1.0.onnx",  # fp32, what media-render runs today
        "model-files-v1.1/kokoro-v1.0.fp16.onnx",  # D4's recommendation
        "model-files-v1.0/kokoro-v1.0.fp16.onnx",  # D4's other candidate, today's graph
    ))


def test_every_kokoro_file_media_render_checks_is_in_the_image_with_the_same_sha256():
    sums = (tool.MEDIA_RENDER / "kokoro" / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    pinned = [line.split() for line in sums if line.strip()]
    carried = {(PurePosixPath(entry["path"]).name, entry["sha256"]) for entry in KOKORO["files"]}
    assert pinned
    for sha256, name in pinned:
        assert (name, sha256) in carried, name


def test_each_kokoro_file_sits_under_its_release_so_the_two_fp16_files_never_collide():
    paths = [asset.path for asset in tool.plan(KOKORO, TRACKS) if asset.path.startswith("kokoro/")]
    assert "kokoro/model-files-v1.0/kokoro-v1.0.fp16.onnx" in paths
    assert "kokoro/model-files-v1.1/kokoro-v1.0.fp16.onnx" in paths
    assert len(set(paths)) == len(paths)
    for entry in KOKORO["files"]:
        assert entry["url"].endswith("/" + entry["path"]), entry["path"]


def test_the_image_takes_exactly_the_tracks_the_music_library_takes():
    music = [asset for asset in tool.plan(KOKORO, TRACKS) if asset.path.startswith("music/")]
    assert [(asset.path, asset.url, asset.sha256) for asset in music] == [
        (f"music/{tool.music.track_file(track)}", track["url"], track["sha256"]) for track in TRACKS
    ]
    assert len(music) == len(json.loads(tool.MUSIC_MANIFEST.read_text(encoding="utf-8"))["tracks"])


# --------------------------------------------------------------------------- the image


def test_the_licence_label_is_apache_and_cc_by():
    assert tool.licences(KOKORO, TRACKS) == "Apache-2.0 AND CC-BY-4.0"


def test_each_licence_is_named_once_in_the_order_it_first_appears():
    tracks = [dict(TRACKS[0], licence="CC0-1.0"), TRACKS[0], dict(TRACKS[0], licence="CC0-1.0")]
    assert tool.licences(KOKORO, tracks) == "Apache-2.0 AND CC0-1.0 AND CC-BY-4.0"


def test_the_notice_carries_the_kokoro_notice_and_every_attribution_line():
    text = tool.notice(KOKORO, TRACKS)
    lines = text.splitlines()
    assert KOKORO["notice"] in lines
    assert "Apache License, Version 2.0" in KOKORO["notice"]
    assert f"/{tool.APACHE_PATH}" in text
    for entry in KOKORO["files"]:
        assert entry["sha256"] in text, entry["path"]
    for track in TRACKS:
        assert track["attribution"] in lines, track["id"]
        assert f"/music/{tool.music.track_file(track)}" in lines


def test_the_licence_text_is_the_apache_terms_without_our_copyright_line():
    text = tool.apache_text(tool.LICENSE)
    assert text.lstrip().startswith("Apache License")
    assert "Version 2.0, January 2004" in text
    assert text.rstrip().endswith("END OF TERMS AND CONDITIONS")
    assert "Automatos" not in text and "APPENDIX" not in text


def test_a_licence_file_without_the_apache_terms_is_refused(tmp_path):
    other = tmp_path / "LICENSE"
    other.write_text("MIT License\n\nPermission is hereby granted...\nEND OF TERMS AND CONDITIONS\n")
    with pytest.raises(tool.AssetsError, match="Apache License 2.0"):
        tool.apache_text(other)


def test_the_image_is_data_only_with_one_layer_per_file():
    assets = tool.plan(KOKORO, TRACKS)
    instructions = [line for line in tool.dockerfile(assets).splitlines() if line and not line.startswith("#")]
    assert instructions[0] == "FROM scratch"
    copies = [line.split() for line in instructions[1:]]
    assert all(len(copy) == 3 and copy[0] == "COPY" and copy[2] == "/" + copy[1] for copy in copies), copies
    assert [copy[1] for copy in copies] == [asset.path for asset in assets] + [tool.APACHE_PATH, tool.NOTICE_PATH]
    assert set(tool.expected_files(KOKORO, TRACKS, "terms")) == {copy[1] for copy in copies}


# --------------------------------------------------------------------------- the Kokoro list

_DROP = object()


def _entry(**changes) -> dict:
    entry = {"path": "model-files-v1.0/voices-v1.0.bin", "url": SOURCE + "model-files-v1.0/voices-v1.0.bin",
             "bytes": 6, "sha256": _sha(b"voices"), "use": "the voices"}
    entry.update(changes)
    return {key: value for key, value in entry.items() if value is not _DROP}


def _list(*files: dict, **changes) -> dict:
    data = {"about": "test", "licence": "Apache-2.0", "notice": "Kokoro, Apache-2.0.", "files": list(files)}
    data.update(changes)
    return {key: value for key, value in data.items() if value is not _DROP}


@pytest.mark.parametrize("data,message", [
    (_list(_entry(url="http://example.test/model-files-v1.0/voices-v1.0.bin")), "url must be an https address"),
    (_list(_entry(url=SOURCE + "model-files-v1.1/voices-v1.0.bin")), "url must end with the path"),
    (_list(_entry(path="../voices-v1.0.bin", url=SOURCE + "../voices-v1.0.bin")), "<release>/<file>"),
    (_list(_entry(path="voices-v1.0.bin", url=SOURCE + "voices-v1.0.bin")), "<release>/<file>"),
    (_list(_entry(bytes=0)), "bytes must be"),
    (_list(_entry(bytes=True)), "bytes must be"),
    (_list(_entry(bytes="6")), "bytes must be"),
    (_list(_entry(sha256="A" * 64)), "sha256 must be"),
    (_list(_entry(use=" ")), "use must say"),
    (_list(_entry(use=_DROP)), "use is required"),
    (_list(_entry(size=6)), "size is not a file field"),
    (_list(_entry(), _entry()), "is listed more than once"),
    (_list(_entry(), licence="MIT"), "licence must be Apache-2.0"),
    (_list(_entry(), notice=" "), "notice must be"),
    (_list(), "files must be a non-empty list"),
    (_list(_entry(), extra=1), "extra is not a list field"),
    (_list(_entry(), about=_DROP), "about is required"),
    ([_entry()], "must be a JSON object"),
])
def test_a_kokoro_list_that_cannot_be_trusted_is_refused(data, message):
    with pytest.raises(tool.AssetsError, match=re.escape(message)):
        tool.validate_kokoro(data)


# --------------------------------------------------------------------------- fetching


class _Opener:
    """Answers every request with the next response (the last one repeats); counts calls."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = 0

    def __call__(self, request, timeout):
        self.calls += 1
        response = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(response, BaseException):
            raise response
        return io.BytesIO(response)


class _ByUrl:
    def __init__(self, payloads: dict):
        self.payloads = payloads

    def __call__(self, request, timeout):
        return io.BytesIO(self.payloads[request.full_url])


def _asset(data: bytes, size=_DROP, path="kokoro/model-files-v1.0/voices-v1.0.bin"):
    return tool.Asset(path, SOURCE + "file", _sha(data), len(data) if size is _DROP else size)


def test_a_file_with_the_pinned_bytes_is_kept_readable_by_all(tmp_path):
    target = tmp_path / "kokoro" / "voices.bin"
    tool.fetch(_asset(b"voices"), target, opener=_Opener(b"voices"), backoff=0)
    assert target.read_bytes() == b"voices"
    assert target.stat().st_mode & 0o777 == 0o644


def test_other_bytes_fail_and_leave_nothing_behind(tmp_path):
    target = tmp_path / "voices.bin"
    with pytest.raises(tool.AssetsError, match="sha256 mismatch"):
        tool.fetch(_asset(b"voices"), target, opener=_Opener(b"voicez"), backoff=0)
    assert not target.exists()


def test_a_kokoro_file_of_another_size_fails(tmp_path):
    target = tmp_path / "voices.bin"
    with pytest.raises(tool.AssetsError, match="6 bytes, where the list pins 7"):
        tool.fetch(_asset(b"voices", size=7), target, opener=_Opener(b"voices"), backoff=0)
    assert not target.exists()


def test_more_bytes_than_pinned_stop_the_download_with_no_retry(tmp_path):
    target, opener = tmp_path / "voices.bin", _Opener(b"voices, and more")
    with pytest.raises(tool.AssetsError, match="sent more than 6 bytes"):
        tool.fetch(_asset(b"voices"), target, opener=opener, backoff=0)
    assert opener.calls == 1 and not target.exists()


def test_a_track_is_capped_at_the_library_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(tool, "TRACK_MAX_BYTES", 4)
    target = tmp_path / "track.mp3"
    with pytest.raises(tool.AssetsError, match="sent more than 4 bytes"):
        tool.fetch(_asset(b"track", size=None, path="music/track.mp3"), target, opener=_Opener(b"track"), backoff=0)
    assert not target.exists()


def test_a_transient_error_is_retried(tmp_path):
    target, opener = tmp_path / "voices.bin", _Opener(ConnectionResetError("reset"), b"voices")
    tool.fetch(_asset(b"voices"), target, opener=opener, backoff=0)
    assert opener.calls == 2 and target.read_bytes() == b"voices"


def test_a_source_that_keeps_failing_fails_after_every_attempt(tmp_path):
    target, opener = tmp_path / "voices.bin", _Opener(urllib.error.URLError("unreachable"))
    with pytest.raises(tool.AssetsError, match=re.escape(f"could not be fetched ({tool.FETCH_ATTEMPTS} attempts)")):
        tool.fetch(_asset(b"voices"), target, opener=opener, backoff=0)
    assert opener.calls == tool.FETCH_ATTEMPTS and not target.exists()


# --------------------------------------------------------------------------- staging and verifying

TRACK = {
    "id": "night-drive", "title": "Night Drive", "artist": "A. Artist", "licence": "CC-BY-4.0",
    "attribution": 'Music: "Night Drive" by A. Artist, licensed CC BY 4.0.',
    "url": "https://music.example.test/night-drive.mp3", "sha256": _sha(b"track"),
}
PAYLOADS = {
    SOURCE + "model-files-v1.0/voices-v1.0.bin": b"voices",
    SOURCE + "model-files-v1.1/kokoro-v1.0.fp16.onnx": b"model",
    TRACK["url"]: b"track",
}


def _sources(tmp_path: Path) -> tuple[Path, Path]:
    """A Kokoro list and a music manifest the real validators take, over PAYLOADS."""
    model = _entry(path="model-files-v1.1/kokoro-v1.0.fp16.onnx", url=SOURCE + "model-files-v1.1/kokoro-v1.0.fp16.onnx",
                   bytes=5, sha256=_sha(b"model"), use="the model")
    kokoro, manifest = tmp_path / "assets.json", tmp_path / "manifest.json"
    kokoro.write_text(json.dumps(_list(_entry(), model)), encoding="utf-8")
    manifest.write_text(json.dumps({"about": "test", "tracks": [TRACK]}), encoding="utf-8")
    return kokoro, manifest


def _build(context: Path, image: Path) -> None:
    """What the staged Dockerfile's COPY lines do: each source to its place in the image."""
    for line in (context / tool.DOCKERFILE).read_text(encoding="utf-8").splitlines():
        if line.startswith("COPY "):
            _, source, destination = line.split()
            target = image / destination.lstrip("/")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(context / source, target)


def _staged_image(tmp_path: Path) -> tuple[Path, dict]:
    kokoro, tracks = tool.load(*_sources(tmp_path))
    licence = tool.apache_text(tool.LICENSE)
    context, image = tmp_path / "context", tmp_path / "image"
    tool.stage_tree(context, kokoro, tracks, licence, opener=_ByUrl(PAYLOADS), backoff=0)
    _build(context, image)
    return image, tool.expected_files(kokoro, tracks, licence)


def test_the_staged_context_builds_exactly_the_image_verify_accepts(tmp_path):
    image, expected = _staged_image(tmp_path)
    assert tool.differences(image, expected) == []
    assert (image / "kokoro" / "model-files-v1.1" / "kokoro-v1.0.fp16.onnx").read_bytes() == b"model"
    assert (image / "music" / "night-drive.mp3").read_bytes() == b"track"
    assert TRACK["attribution"] in (image / "NOTICE").read_text(encoding="utf-8").splitlines()


@pytest.mark.parametrize("tamper,problem", [
    (lambda image: (image / "music" / "night-drive.mp3").write_bytes(b"other"), "/music/night-drive.mp3: sha256"),
    (lambda image: (image / "NOTICE").unlink(), "/NOTICE: missing"),
    (lambda image: (image / "kokoro" / "extra.onnx").write_bytes(b"x"), "/kokoro/extra.onnx: is not one of"),
    (lambda image: (image / "licenses" / "Apache-2.0.txt").write_text("MIT"), "/licenses/Apache-2.0.txt: sha256"),
])
def test_a_published_image_that_differs_in_any_file_fails(tmp_path, tamper, problem):
    image, expected = _staged_image(tmp_path)
    tamper(image)
    problems = tool.differences(image, expected)
    assert len(problems) == 1 and problem in problems[0], problems


def test_staging_refuses_a_directory_that_already_holds_files(tmp_path):
    kokoro, tracks = tool.load(*_sources(tmp_path))
    (tmp_path / "context").mkdir()
    (tmp_path / "context" / "stale.onnx").write_bytes(b"old")
    with pytest.raises(tool.AssetsError, match="must be empty"):
        tool.stage_tree(tmp_path / "context", kokoro, tracks, "terms", opener=_ByUrl(PAYLOADS), backoff=0)


def test_stage_writes_the_licence_expression_and_a_summary(tmp_path, monkeypatch):
    kokoro, manifest = _sources(tmp_path)
    monkeypatch.setattr(tool.urllib.request, "urlopen", _ByUrl(PAYLOADS))
    outputs, summary = tmp_path / "outputs", tmp_path / "summary.md"
    args = ["stage", "--out", str(tmp_path / "context"), "--kokoro", str(kokoro), "--music", str(manifest),
            "--outputs", str(outputs), "--summary", str(summary)]
    assert tool.main(args) == 0
    assert outputs.read_text(encoding="utf-8").splitlines() == ["licences=Apache-2.0 AND CC-BY-4.0"]
    assert "`/music/night-drive.mp3`" in summary.read_text(encoding="utf-8")


def test_stage_fails_the_run_on_a_changed_file(tmp_path, monkeypatch, capsys):
    kokoro, manifest = _sources(tmp_path)
    monkeypatch.setattr(tool.urllib.request, "urlopen", _ByUrl({**PAYLOADS, TRACK["url"]: b"other"}))
    args = ["stage", "--out", str(tmp_path / "context"), "--kokoro", str(kokoro), "--music", str(manifest)]
    assert tool.main(args) == 1
    assert "::error title=media assets::/music/night-drive.mp3: sha256 mismatch" in capsys.readouterr().out
    assert not (tmp_path / "context" / "music" / "night-drive.mp3").exists()


def test_verify_passes_a_matching_tree_and_fails_one_that_differs(tmp_path, capsys):
    image, _ = _staged_image(tmp_path)
    kokoro, manifest = _sources(tmp_path)
    args = ["verify", "--dir", str(image), "--kokoro", str(kokoro), "--music", str(manifest)]
    assert tool.main(args) == 0
    (image / "NOTICE").write_text("changed", encoding="utf-8")
    assert tool.main(args) == 1
    error = capsys.readouterr().out.splitlines()[-1]
    assert error.startswith("::error title=media assets::") and "%0A  /NOTICE: sha256" in error


# --------------------------------------------------------------------------- the job


def test_only_a_manual_run_on_main_that_asks_for_it_publishes_the_image():
    asked = TRIGGERS["workflow_dispatch"]["inputs"]["media_assets"]
    assert asked["type"] == "boolean" and asked["default"] is False
    assert _flat(JOB["if"]) == (
        "github.event_name == 'workflow_dispatch' && github.ref == 'refs/heads/main' && inputs.media_assets == true"
    )
    assert "needs" not in JOB


def _needs(name: str) -> set:
    """Every job ``name`` waits on, directly or through another job."""
    direct = JOBS[name].get("needs") or []
    direct = [direct] if isinstance(direct, str) else direct
    return set(direct).union(*(_needs(need) for need in direct))


# A job whose `if:` calls one of these runs even when a job it needs was skipped.
_RUNS_PAST_A_SKIP = re.compile(r"\b(?:always|cancelled)\(\)")


def test_a_media_assets_run_builds_and_publishes_no_other_image():
    """`changes` skips on such a run; every other image job waits on it, so it skips too."""
    assert JOBS["changes"]["if"] == "inputs.media_assets != true"
    gated = set(JOBS) - {"changes", "media-assets"}
    assert {"build", "report", "merge"} <= gated
    for name in gated:
        assert "changes" in _needs(name), f"{name} would run on a media-assets run"
        condition = str(JOBS[name].get("if", ""))
        assert not _RUNS_PAST_A_SKIP.search(condition), f"{name} would run with its needs skipped: {condition}"


def test_a_media_assets_run_never_replaces_or_waits_behind_a_queued_publish():
    assert IMAGES["concurrency"]["group"] == (
        "${{ github.workflow }}-${{ github.ref }}${{ inputs.media_assets && '-media-assets' || '' }}"
    )


def test_smoke_images_skips_a_media_assets_run_by_its_title():
    title = re.fullmatch(r"\$\{\{ inputs\.media_assets && '([^']+)' \|\| '' \}\}", IMAGES["run-name"]).group(1)
    condition = _flat(SMOKE["jobs"]["smoke-images"]["if"])
    assert f"github.event.workflow_run.display_title != '{title}'" in condition


def test_the_job_checks_the_files_publishes_checks_them_again_and_only_then_signs():
    stage = _step(lambda step: step.get("id") == "stage")
    build = _step(lambda step: step.get("id") == "build")
    verify = _step(lambda step: "media_assets.py verify" in step.get("run", ""))
    sign = _step(lambda step: "cosign sign" in step.get("run", ""))
    order = [JOB["steps"].index(step) for step in (stage, build, verify, sign)]
    assert order == sorted(order)
    assert "scripts/ci/media_assets.py stage" in stage["run"] and '--outputs "$GITHUB_OUTPUT"' in stage["run"]
    assert "for arch in amd64 arm64" in verify["run"]
    assert "printf 'FROM %s@%s\\n' \"$IMAGE\" \"$DIGEST\"" in verify["run"]
    assert "--provenance=false --sbom=false" in verify["run"], "an attestation file would be an extra file"
    assert verify["env"]["DIGEST"] == sign["env"]["DIGEST"] == "${{ steps.build.outputs.digest }}"


def test_the_image_is_built_for_both_platforms_and_pushed_to_ghcr():
    step = _step(lambda step: step.get("id") == "build")
    build = step["with"]
    assert build["context"] == "${{ runner.temp }}/media-assets"
    assert build["platforms"] == "linux/amd64,linux/arm64"
    assert build["outputs"] == "type=image,push=true,rewrite-timestamp=true"
    assert build["tags"] == "${{ steps.meta.outputs.tags }}"
    assert step["env"]["SOURCE_DATE_EPOCH"] == "0"
    assert "cache-to" not in build, "a 700 MB data image stays out of the shared GHA cache"
    name = _step(lambda step: step.get("name") == "Image name")
    assert 'echo "IMAGE=ghcr.io/${GITHUB_REPOSITORY_OWNER,,}/automatos-media-assets"' in name["run"]


def test_its_licence_label_is_the_stage_steps_not_the_apache_2_0_the_image_jobs_stamp():
    meta = _step(lambda step: step.get("id") == "meta")
    for key in ("labels", "annotations"):
        lines = meta["with"][key].splitlines()
        assert "org.opencontainers.image.licenses=${{ steps.stage.outputs.licences }}" in lines
        assert "org.opencontainers.image.licenses=Apache-2.0" not in lines
        assert "org.opencontainers.image.title=automatos-media-assets" in lines
    assert meta["env"]["DOCKER_METADATA_ANNOTATIONS_LEVELS"] == "manifest,index"
    build = _step(lambda step: step.get("id") == "build")["with"]
    assert build["labels"] == "${{ steps.meta.outputs.labels }}"
    assert build["annotations"] == "${{ steps.meta.outputs.annotations }}"
    published = next(step for step in JOBS["build"]["steps"] if step.get("id") == "meta")
    assert meta["uses"] == published["uses"], "the same metadata-action, which sets the revision label"


def test_only_this_job_may_push_and_sign_on_a_media_assets_run():
    assert IMAGES["permissions"] == {"contents": "read"}
    assert JOB["permissions"] == {"contents": "read", "packages": "write", "id-token": "write"}


def test_the_job_reports_the_pin_line_and_whether_an_anonymous_pull_works():
    report = _step(lambda step: "anonymous pull" in step.get("name", ""))
    assert 'echo "FROM ${tag}@${DIGEST} AS assets"' in report["run"]
    assert "https://ghcr.io/token?scope=repository:${repository}:pull" in report["run"]
    assert "::warning title=media-assets is private::" in report["run"]
    assert "exit 1" not in report["run"], "the first publish is always private: report only"


def test_every_action_the_job_uses_is_one_images_yml_already_pins():
    others = {step["uses"] for name, job in JOBS.items() if name != "media-assets"
              for step in job["steps"] if "uses" in step}
    mine = {step["uses"] for step in JOB["steps"] if "uses" in step}
    assert mine and mine <= others, mine - others
