"""Auto's server voice: the voice-tts image definition, its wiring, and the contract both sides hold.

``.github/workflows/voice-tts.yml`` builds the image and proves it (its suite runs inside, and the
running container streams MP3). This suite runs in the required orchestrator job and pins what a
later edit could undo:

- the stages: uv, kokoro (the model and voices checked against SHA256SUMS, the same pins as
  media-render's), py (the hash-checked lock, the espeak data moved to a short real path), runtime
  (lame and tini, no Node, browser or ffmpeg), test, and production last; every base pinned by digest
  and no BuildKit cache mount (Railway refuses one without its service id);
- the lock: every pin of requirements.in kept, every pin hashed, and the versions media-render runs;
- the GPL boundary: kokoro-onnx, espeakng-loader and phonemizer are pinned here, and the orchestrator
  imports nothing from voice_tts (the GPL scan itself is test_prd251w1_media_render_image.py's);
- the contract both sides hold: 500 characters, speed 0.8-1.2, the default voice, and the voice ids
  the core accepts are the ones the service can describe;
- compose (the voice profile, no published port, the backend passing VOICE_TTS_URL and the token
  from .env) and the Railway manifest (the service, its health check, the API's variables).

Pure file reads and voice-tts's standard-library modules: no Docker, no network.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
VOICE_TTS = ROOT / "services" / "voice-tts"
MEDIA_RENDER = ROOT / "services" / "media-render"
DOCKERFILE = (VOICE_TTS / "Dockerfile").read_text(encoding="utf-8")
STAGES = ("uv", "kokoro", "py", "runtime", "test", "production")
_FROM = re.compile(r"^FROM\s+(\S+)\s+AS\s+(\S+)$", re.M)
_PIN = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s;\\]+)", re.M)

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))
# voice-tts's standard-library modules, found LAST so no orchestrator module is shadowed.
if str(VOICE_TTS) not in sys.path:
    sys.path.append(str(VOICE_TTS))

from voice_tts import config as service_config  # noqa: E402
from voice_tts.voices import LANGUAGES, VoicesError, describe  # noqa: E402

from modules.voice import speech  # noqa: E402


def _instructions(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))


def _stage(name: str) -> str:
    match = re.search(rf"^FROM \S+ AS {re.escape(name)}$", DOCKERFILE, re.M)
    assert match, f"no stage named {name}"
    following = re.search(r"^FROM ", DOCKERFILE[match.end():], re.M)
    end = match.end() + following.start() if following else len(DOCKERFILE)
    return _instructions(DOCKERFILE[match.start():end])


def _pins(text: str) -> dict:
    return dict(_PIN.findall(text))


# --------------------------------------------------------------------------- the image


def test_the_stages_are_in_order_with_production_last_and_every_base_pinned():
    stages = _FROM.findall(DOCKERFILE)
    assert [name for _, name in stages] == list(STAGES)
    bases = dict((name, image) for image, name in stages)
    assert bases["test"] == "runtime" and bases["production"] == "runtime"
    for name in ("uv", "kokoro", "py", "runtime"):
        assert re.fullmatch(r"[\w./-]+:[\w.-]+@sha256:[0-9a-f]{64}", bases[name]), name
    assert DOCKERFILE.startswith("# syntax=docker/dockerfile:1\n"), "COPY --link and --start-interval need it"
    assert not re.search(r"--mount=\S*type=cache", _instructions(DOCKERFILE))


def test_the_kokoro_files_are_media_renders_pins_checked_at_build():
    assert (VOICE_TTS / "kokoro" / "SHA256SUMS").read_text() == (MEDIA_RENDER / "kokoro" / "SHA256SUMS").read_text()
    kokoro = _stage("kokoro")
    assert "releases/download/model-files-v1.0" in kokoro
    assert "sha256sum -c SHA256SUMS" in kokoro and "rm SHA256SUMS" in kokoro


def test_the_venv_installs_the_lock_exactly_and_moves_the_espeak_data_to_a_short_real_path():
    py = _stage("py")
    assert "--require-hashes --no-deps" in py and "requirements.lock" in py
    assert re.search(r"mv [^\n]*espeakng_loader[^\n]* /opt/espeak;", py)
    assert "test ! -L /opt/espeak" in py


def test_the_runtime_speaks_only():
    runtime = _stage("runtime")
    assert re.search(r"apt-get install -y --no-install-recommends lame tini;", runtime)
    for absent in ("node", "ffmpeg", "chrome", "exiftool", "curl"):
        assert not re.search(rf"\b{absent}\b", runtime.replace("--no-install-recommends", "")), absent
    assert re.search(r'^ENTRYPOINT \["tini", "--", "python", "-m", "voice_tts"\]$', runtime, re.M)
    assert "/healthz" in runtime and "EXPOSE 8100" in runtime
    assert "USER tts" in runtime


def test_the_lock_keeps_every_input_pin_hashed_at_media_renders_versions():
    wanted = _pins((VOICE_TTS / "requirements.in").read_text())
    lock_text = (VOICE_TTS / "requirements.lock").read_text()
    locked = _pins(lock_text)
    assert {name: locked.get(name) for name in wanted} == wanted
    hashed = re.findall(r"^([A-Za-z0-9_.-]+)==\S+ \\\n    --hash=sha256:[0-9a-f]{64}", lock_text, re.M)
    assert set(hashed) == set(locked), "every pin carries its hashes"
    for name in ("kokoro-onnx", "espeakng-loader", "phonemizer"):
        assert name in wanted, f"the GPL voice stack is pinned here: {name}"
    media_render = _pins((MEDIA_RENDER / "requirements.lock").read_text())
    shared = set(locked) & set(media_render)
    assert {name: locked[name] for name in shared} == {name: media_render[name] for name in shared}
    assert not {"protobuf", "flatbuffers", "c2pa-python"} & set(locked)


def test_the_orchestrator_imports_nothing_from_voice_tts():
    pattern = re.compile(r"^\s*(?:import|from)\s+voice_tts\b", re.M)
    offenders = [
        str(path.relative_to(ROOT)) for path in (ROOT / "orchestrator").rglob("*.py")
        if "tests" not in path.parts and pattern.search(path.read_text(encoding="utf-8", errors="replace"))
    ]
    assert offenders == []


# --------------------------------------------------------------------------- the contract both sides hold


def test_both_sides_hold_the_same_limits_and_default_voice():
    service = service_config.load_settings({}, cpus=1)
    assert speech.MAX_TEXT_CHARS == service.max_chars == service_config.DEFAULT_MAX_CHARS
    assert speech.SPEED_RANGE == service_config.SPEED_RANGE
    assert speech.DEFAULT_SPEED == service_config.DEFAULT_SPEED
    assert service.default_voice == "af_heart"
    source = (_ORCH / "config.py").read_text(encoding="utf-8")
    assert 'os.getenv("VOICE_TTS_DEFAULT_VOICE", "af_heart")' in source


@pytest.mark.parametrize("voice_id", ["af_heart", "bm_george", "ef_dora", "zf_xiaoxiao", "pm_alex"])
def test_a_voice_id_the_core_passes_is_one_the_service_can_describe(voice_id):
    assert speech.VOICE_ID.match(voice_id)
    assert describe(voice_id).id == voice_id


@pytest.mark.parametrize("voice_id", ["ax_heart", "Heart", "af-heart", "af_"])
def test_an_id_outside_kokoros_shape_is_refused_by_both(voice_id):
    assert not speech.VOICE_ID.match(voice_id)
    with pytest.raises(VoicesError):
        describe(voice_id)


def test_a_well_shaped_id_in_no_kokoro_language_passes_the_core_and_the_service_refuses_it():
    """The core checks the shape; the service knows its voices and answers 422, which the core passes on."""
    assert "x" not in LANGUAGES and speech.VOICE_ID.match("xf_heart")
    with pytest.raises(VoicesError):
        describe("xf_heart")


def test_the_readme_states_the_contract_and_the_disclosure_reasoning():
    readme = (VOICE_TTS / "README.md").read_text(encoding="utf-8")
    for needle in ("GET /healthz", "GET /voices", "POST /speak", "tts_warming", "tts_busy", "tts_too_long",
                   "tts_bad_request", "C2PA", "never logged"):
        assert needle in readme, needle


# --------------------------------------------------------------------------- compose and Railway


def test_compose_runs_it_in_the_voice_profile_inside_the_network_only():
    services = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))["services"]
    voice = services["voice-tts"]
    assert voice["profiles"] == ["voice"]
    assert voice["build"]["context"] == "./services/voice-tts" and "image" not in voice
    assert "ports" not in voice and voice["networks"] == ["automatos"]
    assert voice["container_name"] == "automatos_voice_tts"
    assert "8100/healthz" in " ".join(voice["healthcheck"]["test"])
    assert voice["environment"]["VOICE_TTS_TOKEN"] == "${VOICE_TTS_TOKEN:-}"
    backend = services["backend"]["environment"]
    assert backend["VOICE_TTS_URL"] == "${VOICE_TTS_URL:-}" and backend["VOICE_TTS_TOKEN"] == "${VOICE_TTS_TOKEN:-}"
    assert "voice-tts" not in (services["backend"].get("depends_on") or {})


def test_the_railway_manifest_records_the_service_and_the_apis_variables():
    manifest = json.loads((ROOT / "infrastructure" / "railway-manifest.json").read_text(encoding="utf-8"))
    entry = manifest["services"]["voice-tts"]
    assert entry["source"] == {"repo": "AutomatosAI/automatos-ai", "root_dir": "/services/voice-tts"}
    assert entry["builder"] == "DOCKERFILE" and entry["port"] == 8100 and entry["healthcheck"] == "/readyz"
    assert entry["replicas"] == 1 and entry["env_keys"] == sorted(entry["env_keys"])
    assert "VOICE_TTS_TOKEN" in entry["env_keys"]
    assert "voice-tts" in manifest["service_groups"]["core"]["services"]
    assert "voice-tts" in manifest["network_topology"]["internal_only"]
    assert {"VOICE_TTS_URL", "VOICE_TTS_TOKEN"} <= set(manifest["services"]["automatos-ai-api"]["env_keys"])
