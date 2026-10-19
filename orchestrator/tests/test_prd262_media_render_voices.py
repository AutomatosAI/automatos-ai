"""PRD-262 US-052: the renderer's language data matches what the product speaks.

media-render.yml builds the image and proves it (the proof run: the in-image suite
and the refusal step). This suite runs in the required orchestrator job and pins
what a later edit could undo:

- voices.json is the catalogue: the ids of ``modules/socials/kokoro_voices.py``, in
  its order. The media-render build context cannot read the orchestrator, so the ids
  are copied there, and this test keeps the two equal;
- the renderer speaks the two languages the orchestrator sends, both from en_dict;
- the kokoro stage rewrites voices-v1.0.bin to the catalogue after the upstream
  sha256 check, and ships no SHA256SUMS that no longer describes it
  (media_render/voices_build.py: byte for byte, in upstream's order, the same input
  giving the same file, and a catalogue voice upstream lacks fails the build);
- the py stage keeps en_dict alone among espeak-ng's dictionaries, in the RUN that
  installs the lock and moves the data, and fails the build when the cut changes
  Kokoro's phonemes for the fixed sentence (media_render/espeak_build.py);
- validation follows the data: a voice is one the voices file holds, a language is
  en-us or en-gb, each refusal names its field, and MEDIA_RENDER_KOKORO_LANG outside
  the two stops the boot;
- the proof run: every catalogue voice speaks a line, the phoneme snapshot holds,
  and the running image answers 400 for a voice outside voices.json and a language
  other than en-us or en-gb.

File reads, temporary files and media-render's standard-library modules: no Docker,
no network, no espeak-ng.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from tests.test_prd262_media_render_stages import DOCKERFILE, MEDIA_RENDER, _stage, _step, _steps

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))
# media-render's modules (standard library only), found LAST so no orchestrator module is shadowed.
if str(MEDIA_RENDER) not in sys.path:
    sys.path.append(str(MEDIA_RENDER))

from media_render import espeak_build, validate, voices_build  # noqa: E402
from media_render.bundle import parse_bundle  # noqa: E402
from media_render.config import ConfigError, load_settings  # noqa: E402
from media_render.fixture import fixture_bundle  # noqa: E402
from media_render.validate import BundleError  # noqa: E402
from modules.socials.kokoro_voices import AMERICAN, BRITISH, KOKORO_VOICES, language_of  # noqa: E402

CATALOGUE_IDS = tuple(voice_id for voice_id, _name, _description in KOKORO_VOICES)
VOICES = frozenset(CATALOGUE_IDS)
# In upstream's voices-v1.0.bin and not in the catalogue: an English voice and a Chinese one.
OUTSIDE = ("am_santa", "zf_xiaoxiao")
STEP = "A voice outside voices.json and a language other than en-us or en-gb are each a 400 naming the field"


# ── the catalogue and the languages ─────────────────────────────────────────


def test_voices_json_is_the_orchestrators_catalogue():
    assert voices_build.CATALOGUE == MEDIA_RENDER / "media_render" / "voices.json"
    assert voices_build.load_catalogue() == CATALOGUE_IDS
    assert len(CATALOGUE_IDS) == 27 and not set(OUTSIDE) & VOICES


def test_the_renderer_speaks_the_two_languages_the_orchestrator_sends():
    assert espeak_build.LANGUAGES == (AMERICAN, BRITISH)
    assert {language_of(voice_id) for voice_id in CATALOGUE_IDS} == set(espeak_build.LANGUAGES)
    # espeak-ng reads both from one dictionary.
    assert espeak_build.dictionaries_for() == {"en_dict"}


# ── the image definition ────────────────────────────────────────────────────


def test_the_kokoro_stage_rewrites_the_voices_after_the_upstream_check():
    kokoro = _stage("kokoro")
    copied = kokoro.index("COPY --link media_render/voices.json media_render/voices_build.py /build/voices/")
    (run,) = [line for line in kokoro.splitlines() if line.startswith("RUN ")]
    assert copied < kokoro.index(run)
    check = run.index("cd /opt/kokoro && sha256sum -c SHA256SUMS")
    rewrite = run.index("python /build/voices/voices_build.py /build/voices/voices.json voices-v1.0.bin")
    # The upstream sums no longer describe the rewritten file, so they stay in this stage.
    assert check < rewrite < run.index("rm SHA256SUMS")
    assert " && " in run[check:rewrite] and " && " in run[rewrite:]
    assert "COPY --link --from=kokoro /opt/kokoro /opt/kokoro" in _stage("runtime")


def test_the_py_stage_keeps_en_dict_alone_in_the_run_that_installs_the_lock():
    py = _stage("py")
    copied = py.index("COPY media_render/espeak_build.py /opt/media-render/build/espeak_build.py")
    (run,) = [line for line in py.splitlines() if line.startswith("RUN ") and "uv pip install" in line]
    assert copied < py.index(run)
    install = run.index("uv pip install")
    move = run.index("/opt/espeak;")
    cut = run.index(
        "/opt/venv/bin/python /opt/media-render/build/espeak_build.py /opt/espeak /opt/media-render/espeak-phonemes.json;"
    )
    assert install < move < cut < run.index("chmod -R a+rX /opt/espeak;")
    # The snapshot goes to the test stage alone; production carries nothing extra.
    snapshot = "COPY --from=py /opt/media-render/espeak-phonemes.json /opt/media-render/espeak-phonemes.json"
    assert snapshot in _stage("test")
    assert "espeak-phonemes.json" not in _stage("runtime") and "espeak-phonemes.json" not in _stage("production")


# ── voices_build: the voices file rewritten to the catalogue ────────────────


def _voices_file(path: Path, voices) -> dict:
    """A voices file laid out as upstream's: a stored zip, one <voice>.npy each, dated 1980."""
    members = {voice: f"npy bytes of {voice}".encode() * 64 for voice in voices}
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as archive:
        for voice, data in members.items():
            archive.writestr(zipfile.ZipInfo(f"{voice}.npy", date_time=(1980, 1, 1, 0, 0, 0)), data)
    os.utime(path, (1_700_000_000, 1_700_000_000))
    return members


def test_a_voices_file_is_rewritten_to_the_catalogue_byte_for_byte(tmp_path):
    upstream = ("af_heart", "am_santa", "bf_emma", "jf_alpha", "zf_xiaoxiao")
    first, second = tmp_path / "a.bin", tmp_path / "b.bin"
    members = _voices_file(first, upstream)
    _voices_file(second, upstream)
    done = voices_build.rewrite(first, ["bf_emma", "af_heart"])
    assert (done.held, done.kept) == (5, 2) and done.bytes_after < done.bytes_before
    assert voices_build.voice_names(first) == {"af_heart", "bf_emma"}
    with zipfile.ZipFile(first) as archive:
        # Upstream's order, each member's bytes and date as they were.
        assert archive.namelist() == ["af_heart.npy", "bf_emma.npy"]
        assert all(archive.read(f"{voice}.npy") == members[voice] for voice in ("af_heart", "bf_emma"))
        assert {info.date_time for info in archive.infolist()} == {(1980, 1, 1, 0, 0, 0)}
        assert {info.compress_type for info in archive.infolist()} == {zipfile.ZIP_STORED}
    assert first.stat().st_mtime == 1_700_000_000
    # The same input gives the same file, whatever order the catalogue lists.
    voices_build.rewrite(second, ["af_heart", "bf_emma"])
    assert first.read_bytes() == second.read_bytes()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["a.bin", "b.bin"]


def test_a_catalogue_voice_the_upstream_file_lacks_fails_the_build(tmp_path):
    voices = tmp_path / "voices-v1.0.bin"
    _voices_file(voices, ("af_heart", "am_santa"))
    before = voices.read_bytes()
    with pytest.raises(voices_build.VoicesError, match="has no bf_emma"):
        voices_build.rewrite(voices, ["af_heart", "bf_emma"])
    assert voices.read_bytes() == before
    catalogue = tmp_path / "voices.json"
    catalogue.write_text(json.dumps({"voices": ["af_heart", "bf_emma"]}))
    assert voices_build.main([str(catalogue), str(voices)]) == 1
    catalogue.write_text(json.dumps({"voices": ["af_heart"]}))
    assert voices_build.main([str(catalogue), str(voices)]) == 0
    assert voices_build.voice_names(voices) == {"af_heart"}


@pytest.mark.parametrize(
    "content, message",
    [
        ("not json", "cannot be read"),
        (json.dumps({"voices": []}), "must list the voice ids"),
        (json.dumps({"voices": ["af_heart", "Heart"]}), "not Kokoro voice ids"),
        (json.dumps({"voices": ["af_heart", "af_heart"]}), "more than once"),
    ],
)
def test_a_catalogue_that_cannot_be_used_fails_the_build(tmp_path, content, message):
    catalogue = tmp_path / "voices.json"
    catalogue.write_text(content)
    with pytest.raises(voices_build.VoicesError, match=message):
        voices_build.load_catalogue(catalogue)


def test_a_file_that_is_not_a_voices_file_is_named(tmp_path):
    text = tmp_path / "text.bin"
    text.write_text("not a zip")
    for path in (tmp_path / "absent.bin", text):
        with pytest.raises(voices_build.VoicesError, match=f"the Kokoro voices file {re.escape(str(path))} cannot be read"):
            voices_build.voice_names(path)
    empty = tmp_path / "empty.bin"
    with zipfile.ZipFile(empty, "w") as archive:
        archive.writestr("README.txt", "no voices")
    with pytest.raises(voices_build.VoicesError, match="holds no voices"):
        voices_build.voice_names(empty)


# ── espeak_build: the dictionaries cut to the languages' ────────────────────


def _espeak_data(root: Path) -> Path:
    data = root / "espeak"
    (data / "lang" / "gmw").mkdir(parents=True)
    (data / "voices" / "!v").mkdir(parents=True)
    (data / "lang" / "gmw" / "en-US").write_text("name English (America)")
    for name in ("en_dict", "fr_dict", "ru_dict", "phontab", "phondata", "phonindex", "intonations"):
        (data / name).write_bytes(name.encode() * 10)
    return data


def test_every_dictionary_but_en_dict_is_deleted(tmp_path):
    data = _espeak_data(tmp_path)
    dropped, size = espeak_build.prune(data, espeak_build.dictionaries_for())
    assert dropped == ("fr_dict", "ru_dict") and size == len(b"fr_dict" * 10) + len(b"ru_dict" * 10)
    assert sorted(path.name for path in data.iterdir()) == [
        "en_dict", "intonations", "lang", "phondata", "phonindex", "phontab", "voices",
    ]
    (data / "en_dict").unlink()
    with pytest.raises(espeak_build.EspeakBuildError, match="has no en_dict"):
        espeak_build.prune(data, espeak_build.dictionaries_for())


def test_the_build_writes_the_snapshot_when_the_phonemes_hold(tmp_path, monkeypatch):
    data, snapshot = _espeak_data(tmp_path), tmp_path / "out" / "espeak-phonemes.json"
    taken = {"en-us": "ɔn θɹˈiː mˈɑːɹtʃ", "en-gb": "ɒn θɹˈiː mˈɑːtʃ"}
    seen = []

    def phonemes(path):
        seen.append(sorted(p.name for p in path.glob("*_dict")))
        return dict(taken)

    monkeypatch.setattr(espeak_build, "phonemes", phonemes)
    record = espeak_build.build(data, snapshot)
    # Taken with the full data, then again after the cut.
    assert seen == [["en_dict", "fr_dict", "ru_dict"], ["en_dict"]]
    written = json.loads(snapshot.read_text(encoding="utf-8"))
    assert written == record and written["phonemes"] == taken and list(written["phonemes"]) == ["en-us", "en-gb"]
    assert written["sentence"] == espeak_build.SENTENCE and written["dictionaries"] == ["en_dict"]


def test_the_build_fails_when_the_cut_changes_kokoros_phonemes(tmp_path, monkeypatch):
    data = _espeak_data(tmp_path)
    answers = iter([{"en-us": "kafeɪ", "en-gb": "kafeɪ"}, {"en-us": "kafeɪ", "en-gb": "kˈæfeɪ"}])
    monkeypatch.setattr(espeak_build, "phonemes", lambda path: next(answers))
    with pytest.raises(espeak_build.EspeakBuildError, match="changed Kokoro's phonemes .en-gb: 'kafeɪ' -> 'kˈæfeɪ'"):
        espeak_build.build(data, tmp_path / "espeak-phonemes.json")
    assert not (tmp_path / "espeak-phonemes.json").exists()


@pytest.mark.parametrize(
    "returncode, stdout, message",
    [
        (1, "", "failed .exit 1."),
        (0, "Traceback-free noise\n", "printed no phonemes"),
        (0, json.dumps({"en-us": "ɐ", "en-gb": ""}) + "\n", "gave no phonemes in en-gb"),
    ],
)
def test_a_child_that_cannot_phonemize_fails_the_build(tmp_path, monkeypatch, returncode, stdout, message):
    def run(argv, **kwargs):
        assert argv[1:] == [str(Path(espeak_build.__file__).resolve()), "--phonemize", str(tmp_path)]
        assert kwargs["timeout"] == espeak_build.PHONEMIZE_TIMEOUT_SECONDS and kwargs["check"] is False
        return subprocess.CompletedProcess(argv, returncode, stdout=stdout, stderr="")

    monkeypatch.setattr(espeak_build.subprocess, "run", run)
    with pytest.raises(espeak_build.EspeakBuildError, match=message):
        espeak_build.phonemes(tmp_path)


def test_the_child_prints_its_phonemes_as_ascii_json(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(espeak_build, "_phonemize_here", lambda path: {"en-us": "kˈæfeɪ", "en-gb": "kˈafeɪ"})
    assert espeak_build.main([str(tmp_path), "--phonemize"]) == 0
    printed = capsys.readouterr().out
    assert printed.isascii() and json.loads(printed) == {"en-us": "kˈæfeɪ", "en-gb": "kˈafeɪ"}


# ── validation follows the data ─────────────────────────────────────────────


def test_no_voice_or_language_pattern_is_left():
    assert not hasattr(validate, "VOICE_NAME") and not hasattr(validate, "LANGUAGE")
    source = (MEDIA_RENDER / "media_render" / "validate.py").read_text()
    assert "from .espeak_build import LANGUAGES" in source


@pytest.mark.parametrize("voice", [*OUTSIDE, "AF_HEART", "af_", None, 3])
def test_a_voice_outside_the_voices_file_is_refused_naming_the_field(voice):
    with pytest.raises(BundleError, match=r"^audio\.voice\.voice must be one of the renderer's Kokoro voices: af_alloy, af_aoede, "):
        validate.voice_name(voice, "audio.voice.voice", VOICES)
    assert validate.voice_name("bm_george", "voice", VOICES) == "bm_george"


@pytest.mark.parametrize("lang", ["fr-fr", "en", "en-US", "zh", "", None])
def test_a_language_other_than_en_us_or_en_gb_is_refused_naming_the_field(lang):
    with pytest.raises(BundleError, match=r"^lang must be one of en-us, en-gb$"):
        validate.language(lang, "lang")
    assert [validate.language(code, "lang") for code in ("en-us", "en-gb")] == ["en-us", "en-gb"]


def test_a_render_bundle_names_the_voice_and_the_language_it_was_refused_for():
    settings = load_settings({})
    body = fixture_bundle()
    spoken = body["audio"]["voice"]
    for changes, field in (({"voice": "am_santa"}, "audio.voice.voice"), ({"lang": "fr-fr"}, "audio.voice.lang")):
        refused = {**body, "audio": {**body["audio"], "voice": {**spoken, **changes}}}
        with pytest.raises(BundleError, match=rf"^{re.escape(field)} must be one of "):
            parse_bundle(refused, settings, {}, VOICES)
    british = {**body, "audio": {**body["audio"], "voice": {**spoken, "voice": "bf_emma", "lang": "en-gb"}}}
    assert parse_bundle(british, settings, {}, VOICES).audio.voice.lang == "en-gb"


def test_a_language_the_image_does_not_speak_stops_the_boot():
    for lang in ("fr-fr", "en"):
        with pytest.raises(ConfigError, match=re.escape(f"MEDIA_RENDER_KOKORO_LANG={lang!r} is not valid: expected one of en-us, en-gb")):
            load_settings({"MEDIA_RENDER_KOKORO_LANG": lang})
    assert load_settings({}).kokoro_lang == "en-us"
    assert load_settings({"MEDIA_RENDER_KOKORO_LANG": "en-gb"}).kokoro_lang == "en-gb"


def test_boot_reads_the_voices_file_and_the_app_parses_with_its_names():
    boot = (MEDIA_RENDER / "media_render" / "boot.py").read_text()
    assert "*kokoro_voice_problems(settings)," in boot
    assert "voices = voice_names(settings.kokoro_voices_path)" in boot
    server = (MEDIA_RENDER / "media_render" / "server.py").read_text()
    assert "voices=voice_names(settings.kokoro_voices_path) if voices is None else frozenset(voices)," in server
    assert "parse_bundle(payload, state.settings, state.library, state.voices)" in server
    assert "parse_tts_request(payload, state.settings, state.voices)" in server


# ── the proof run ───────────────────────────────────────────────────────────


def test_the_in_image_suite_speaks_every_voice_and_holds_the_snapshot():
    kokoro = (MEDIA_RENDER / "tests" / "test_kokoro.py").read_text()
    for needle in (
        '@pytest.mark.parametrize("voice", load_catalogue())',
        "def test_every_catalogue_voice_speaks_a_line(voice, tmp_path):",
        'lang = "en-gb" if voice.startswith("b") else "en-us"',
        'PHONEMES = Path("/opt/media-render/espeak-phonemes.json")',
        'assert tokenizer.phonemize(snapshot["sentence"], language) == expected, language',
    ):
        assert needle in kokoro, needle
    image = (MEDIA_RENDER / "tests" / "test_image.py").read_text()
    for needle in (
        "assert voice_names(settings.kokoro_voices_path) == frozenset(load_catalogue())",
        'assert sorted(path.name for path in data.glob("*_dict")) == ["en_dict"]',
    ):
        assert needle in image, needle
    # The media-render job runs that suite inside the image.
    assert '--entrypoint python "$TEST_IMAGE" -m pytest' in _step("media-render", "--target test")


def test_the_proof_run_gets_a_400_naming_the_field_from_the_running_image():
    names = [step.get("name") for step in _steps("media-render")]
    assert STEP in names and names.index(STEP) < len(names) - 1, "the size check stays last"
    step = _step("media-render", "assert_voice_refusals.py")
    assert '"$IMAGE" fixture-bundle > "$out/bundle.json"' in step
    assert "-e SOCIALS_RENDER_TOKEN=\"$TOKEN\"" in step
    assert "--voices services/media-render/media_render/voices.json" in step
    assert 'exit "$code"' in step
    helper = (MEDIA_RENDER / "ci" / "assert_voice_refusals.py").read_text()
    assert 'OUTSIDE_VOICES = ("am_santa", "zf_xiaoxiao")' in helper and 'OTHER_LANGUAGE = "fr-fr"' in helper
    for field in ('"voice")', '"lang")', '"audio.voice.voice")', '"audio.voice.lang")'):
        assert field in helper, field
    assert 'message.startswith(f"{field} must be one of ")' in helper


def _refusals():
    spec = importlib.util.spec_from_file_location("assert_voice_refusals", MEDIA_RENDER / "ci" / "assert_voice_refusals.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _image_answers(route: str, body: dict) -> tuple:
    """What the image answers: the parser's own refusal, or the spoken line."""
    settings = load_settings({})
    try:
        if route == "/render":
            parse_bundle(body, settings, {}, VOICES)
        else:
            validate.voice_name(body.get("voice", settings.kokoro_voice), "voice", VOICES)
            validate.language(body.get("lang", settings.kokoro_lang), "lang")
    except BundleError as exc:
        return 400, {"error": "invalid_bundle" if route == "/render" else "invalid_request", "message": str(exc)}
    return 200, {"voice": body.get("voice"), "lang": body.get("lang"), "lines": []}


def test_the_refusal_step_passes_on_the_images_answers_and_fails_on_anything_else(monkeypatch):
    helper = _refusals()
    bundle = fixture_bundle()
    monkeypatch.setattr(helper, "_post", lambda url, token, body, timeout: _image_answers(url[len("http://x"):], body))
    assert helper.check("http://x", "t", bundle, list(CATALOGUE_IDS), 5.0) == []
    # A renderer that took any voice (the pattern before US-052) fails every voice case.
    monkeypatch.setattr(helper, "_post", lambda url, token, body, timeout: (200, {"voice": body.get("voice"), "lang": body.get("lang")}))
    failures = helper.check("http://x", "t", bundle, list(CATALOGUE_IDS), 5.0)
    assert len(failures) == 6 and all("answered 200" in failure for failure in failures)
    # A voice the catalogue holds proves nothing.
    assert helper.check("http://x", "t", bundle, ["am_santa"], 5.0)[0].startswith("am_santa is in voices.json")


def test_the_dockerfile_says_where_the_cut_is_made():
    for needle in ("PRD-262 US-052", "media_render/voices_build.py", "only en_dict stays"):
        assert needle in DOCKERFILE, needle
