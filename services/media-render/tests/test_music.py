"""The music library (PRD-251 S1.6, US-112): built at image build, read at boot.

- The committed manifest lists every track with its licence, attribution, source
  URL and sha256, and the build refuses one that does not.
- A download whose sha256 differs from the manifest's fails the build (the
  media-render CI job also builds the image from a manifest with a bad hash and
  requires the build to fail).
- The analysis finds breaks, breakdowns, drops and grooves at 0.1 s: proved on a
  synthetic track whose structure is known, decoded by the image's own ffmpeg.
- The image carries the four reference tracks, each hash-checked, with cue
  windows, and Deep House 003's break at 34.0-35.8 s (the Markets reference's cut).
- A render that mixes a track reports it, with the licence and the credit line.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import urllib.error
import wave
from pathlib import Path

import numpy as np
import pytest

from media_render import music_build
from media_render.bundle import parse_bundle
from media_render.config import load_settings
from media_render.fixture import BUNDLES, SCRIPT_COMPOSITION, music_bundle
from media_render.music import MusicLibraryError, load_library
from media_render.music_build import LICENCES, MusicBuildError
from media_render.pipeline import music_report

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "music" / "manifest.json"
# The four tracks the reference videos used (verified byte-identical 2026-09-23).
REFERENCE_TRACKS = {
    "where-the-night-begins": ("b171474b-0ac5-4daa-a012-28fea778fb86", "2feeff3d7bfffc24712bf0b9df1efc76a1777973fde3143b061579c4541d6c83"),
    "da-da-da-da-da-de-de-de-de": ("ec7b18f5-b8d6-4438-9c8e-081f531564e0", "5e71ada9dc33cd4fc21e69a38008b4b263d45de8aca2b2b5ed466816f66e9792"),
    "spring-of-2026": ("d130cacc-e8a6-457f-ac48-e21aa9bd67c5", "ab1805743e5c3a728013ad45e06a80fe6cd22384ea9f2602e4d9209cf585016d"),
    "deep-house-003": ("e1168d5d-f775-4a55-856a-dbbf62d115a5", "ddb3174cb4f2337a04c918902e4a766a427e947c6b04ae8d406317db07b32109"),
}
MARKETS_BREAK = (34.0, 35.8)
RATE = music_build.ANALYSIS_RATE
TOLERANCE = 0.21
# A groove beside a busy mid-range stretch stops short of it by the mid share's smoothing (0.5 s at most).
GROOVE_TOLERANCE = 0.51


def _source() -> dict:
    return json.loads(SOURCE.read_text())


# ── the committed manifest ──────────────────────────────────────────────────
def test_every_committed_track_has_licence_attribution_url_and_sha256():
    tracks = music_build.load_source(SOURCE)
    assert {t["id"]: (t["url"].rsplit("/", 1)[-1].split(".")[0], t["sha256"]) for t in tracks} == REFERENCE_TRACKS
    for track in tracks:
        assert track["url"].startswith("https://ende.app/storage/mp3low/") and track["url"].endswith(".mp3")
        assert (track["artist"], track["licence"]) == ("Sascha Ende", "CC-BY-4.0")
        assert track["attribution"] == f'Music: "{track["title"]}" by Sascha Ende (ende.app), licensed CC BY 4.0.'
        assert music_build.track_file(track) == f"{track['id']}.mp3"


def test_no_audio_file_is_committed_beside_the_manifest():
    assert sorted(p.name for p in SOURCE.parent.iterdir()) == ["manifest.json"]


@pytest.mark.parametrize(
    ("change", "problem"),
    [
        (lambda t: t.update(sha256="A" * 64), "sha256 must be 64 lowercase hex"),
        (lambda t: t.update(licence="All rights reserved"), "licence must be one of"),
        (lambda t: t.pop("attribution"), "attribution is required"),
        (lambda t: t.update(attribution="Music by a friend"), "the attribution must name the title"),
        (lambda t: t.update(attribution=f'"{t["title"]}" by {t["artist"]}'), "must name its licence"),
        (lambda t: t.update(url="http://ende.app/track.mp3"), "url must be an https address"),
        (lambda t: t.update(url="https://ende.app/track"), "url must name an audio file"),
        (lambda t: t.update(id="Deep House"), "id must be lowercase words"),
        (lambda t: t.update(bpm=124), "is not a track field"),
    ],
)
def test_the_build_refuses_a_track_it_could_not_credit_or_pin(change, problem):
    data = _source()
    change(data["tracks"][0])
    with pytest.raises(MusicBuildError) as refused:
        music_build.validate_source(data)
    assert problem in str(refused.value)


def test_a_track_id_is_used_once():
    data = _source()
    data["tracks"].append(copy.deepcopy(data["tracks"][0]))
    with pytest.raises(MusicBuildError, match="is used by tracks\\[0\\] too"):
        music_build.validate_source(data)


# ── fetching and the hash ───────────────────────────────────────────────────
class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _serving(files, failures=0):
    """An opener serving ``files`` by URL; the first ``failures`` calls fail like a dropped connection."""
    calls = []

    def opener(request, timeout):
        calls.append(request.full_url)
        if len(calls) <= failures:
            raise urllib.error.URLError("connection reset")
        return _Response(files[request.full_url])

    opener.calls = calls
    return opener


def _one_track_source(tmp_path, data: bytes, sha256: str) -> Path:
    entry = {
        "id": "test-groove", "title": "Test Groove", "artist": "The Suite", "licence": "CC-BY-4.0",
        "attribution": 'Music: "Test Groove" by The Suite, licensed CC BY 4.0.',
        "url": "https://music.example.test/test-groove.wav", "sha256": sha256,
    }
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"tracks": [entry]}))
    return source


def test_a_download_whose_sha256_differs_fails_the_build(tmp_path, capsys, monkeypatch):
    data = _wav_bytes(_synthetic_track())
    source = _one_track_source(tmp_path, data, hashlib.sha256(b"another file").hexdigest())
    opener = _serving({"https://music.example.test/test-groove.wav": data})
    with pytest.raises(MusicBuildError, match="test-groove: sha256 mismatch"):
        music_build.build(source, tmp_path / "library", opener=opener)
    assert not (tmp_path / "library" / "manifest.json").exists()
    # The Dockerfile runs main(): a mismatch is a non-zero exit, so the image build stops.
    monkeypatch.setattr(music_build.urllib.request, "urlopen", opener)
    code = music_build.main([str(source), str(tmp_path / "library")])
    assert code == 1 and "sha256 mismatch" in capsys.readouterr().err
    assert not (tmp_path / "library" / "manifest.json").exists()


def test_a_dropped_transfer_is_retried_and_one_that_never_arrives_fails(tmp_path):
    target = tmp_path / "t.mp3"
    url = "https://music.example.test/t.mp3"
    flaky = _serving({url: b"ID3 some audio"}, failures=2)
    digest = music_build.fetch(url, target, opener=flaky, attempts=3, backoff=0)
    assert digest == hashlib.sha256(b"ID3 some audio").hexdigest() and len(flaky.calls) == 3
    with pytest.raises(MusicBuildError, match="could not be fetched \\(3 attempts\\)"):
        music_build.fetch(url, target, opener=_serving({url: b"x"}, failures=3), attempts=3, backoff=0)


# ── the analysis, on a track whose structure is known ───────────────────────
def _synthetic_track() -> np.ndarray:
    """30 s at 22.05 kHz: a hi-hat throughout; the bass in from 3.0 s, out 10.0-12.0 s (a break),
    out 18.0-23.0 s under a loud mid-range line (a breakdown), and gone for good at 28.0 s (an outro)."""
    t = np.arange(int(30 * RATE)) / RATE
    hat = 0.05 * np.sin(2 * np.pi * 5000 * t)
    bass_in = (t >= 3.0) & ~((t >= 10.0) & (t < 12.0)) & ~((t >= 18.0) & (t < 23.0)) & (t < 28.0)
    bass = 0.3 * np.sin(2 * np.pi * 80 * t) * bass_in
    vocal = 0.3 * np.sin(2 * np.pi * 1000 * t) * ((t >= 18.0) & (t < 23.0))
    return (hat + bass + vocal).astype(np.float32)


def _wav_bytes(samples: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    return buffer.getvalue()


def _near(actual, expected):
    return abs(actual - expected) <= TOLERANCE


def _assert_known_structure(cues):
    (brk,) = cues["breaks"]
    assert _near(brk["start"], 10.0) and _near(brk["end"], 12.0), brk
    assert brk["depth_db"] > 20
    (breakdown,) = cues["breakdowns"]
    assert _near(breakdown["start"], 18.0) and _near(breakdown["end"], 23.0), breakdown
    assert breakdown["mid_share"] > 0.5, "the breakdown is the loud mid-range line"
    # The beat coming in after the intro is a drop too; the outro has none.
    assert len(cues["drops"]) == 3 and all(_near(a, e) for a, e in zip(cues["drops"], (3.0, 12.0, 23.0))), cues["drops"]
    grooves = [(g["start"], g["end"]) for g in cues["grooves"]]
    expected = ((3.0, 10.0), (12.0, 18.0), (23.0, 28.0))
    assert len(grooves) == 3 and all(
        abs(s - es) <= GROOVE_TOLERANCE and abs(e - ee) <= GROOVE_TOLERANCE for (s, e), (es, ee) in zip(grooves, expected)
    ), grooves
    # ...and never inside the breakdown's busy mid range.
    assert all(e <= 18.0 or s >= 23.0 for s, e in grooves), grooves


def test_the_analysis_finds_breaks_breakdowns_drops_and_grooves_at_a_tenth_of_a_second():
    frames = music_build.frame_features(_synthetic_track(), RATE)
    assert len(frames) == 300 and frames[1].at == 0.1
    _assert_known_structure(music_build.find_cues(frames))
    every = [t for kind in ("breaks", "breakdowns", "grooves") for w in music_build.find_cues(frames)[kind] for t in (w["start"], w["end"])]
    assert all(round(t, 1) == t for t in every), "cue times are at 0.1 s"


def test_the_map_gives_the_three_measures_every_two_seconds():
    frames = music_build.frame_features(_synthetic_track(), RATE)
    mapped = music_build.loudness_map(frames)
    assert mapped["step"] == 2.0 and len(mapped["loudness_db"]) == len(mapped["bass_db"]) == len(mapped["mid_share"]) == 15
    assert mapped["bass_db"][2] - mapped["bass_db"][5] > 20, "4-6 s has the bass, 10-12 s does not"
    assert mapped["mid_share"][10] > 0.5 > mapped["mid_share"][7], "18-20 s is the mid-range line"


def test_a_built_track_is_fetched_verified_decoded_by_ffmpeg_and_analysed(tmp_path):
    data = _wav_bytes(_synthetic_track())
    source = _one_track_source(tmp_path, data, hashlib.sha256(data).hexdigest())
    library = tmp_path / "library"
    manifest = music_build.build(source, library, opener=_serving({"https://music.example.test/test-groove.wav": data}))
    (track,) = manifest["tracks"]
    assert (library / "test-groove.wav").read_bytes() == data
    assert track["file"] == "test-groove.wav" and abs(track["duration"] - 30.0) < 0.05
    assert (track["licence_name"], track["licence_url"], track["credit_required"]) == (
        "CC BY 4.0", LICENCES["CC-BY-4.0"]["url"], True,
    )
    _assert_known_structure(track["cues"])
    assert json.loads((library / "manifest.json").read_text()) == manifest
    # The service loads what the build wrote.
    loaded = load_library(str(library))["test-groove"]
    assert loaded.credit_required and loaded.report()["attribution"] == 'Music: "Test Groove" by The Suite, licensed CC BY 4.0.'


# ── the library the image carries ───────────────────────────────────────────
def _image_library():
    return load_library(load_settings().music_dir)


def test_the_image_carries_every_committed_track_hash_checked_with_cue_windows():
    library = _image_library()
    built = json.loads((Path(load_settings().music_dir) / "manifest.json").read_text())
    assert sorted(library) == sorted(REFERENCE_TRACKS)
    for track_id, (_, sha256) in REFERENCE_TRACKS.items():
        track = library[track_id]
        assert hashlib.sha256(track.path.read_bytes()).hexdigest() == sha256
        assert track.duration and track.duration > 60
        assert track.licence == "CC-BY-4.0" and track.credit_required
        assert music_build.cue_windows(track.cues) > 0, f"{track_id} has no cue windows"
    for entry in built["tracks"]:
        assert entry["licence_name"] == "CC BY 4.0" and entry["attribution"] and entry["url"] and entry["sha256"]
        assert entry["map"]["step"] == 2.0 and entry["map"]["bass_db"]


def test_deep_house_003_lists_the_break_the_markets_reference_cut_on():
    breaks = _image_library()["deep-house-003"].cues["breaks"]
    low, high = MARKETS_BREAK
    assert any(w["start"] < high and w["end"] > low for w in breaks), breaks


def test_every_reference_window_fits_its_track():
    # (track, start, length): the four reference videos' music windows (the seeded templates').
    windows = [
        ("where-the-night-begins", 190.1, 39.5),
        ("da-da-da-da-da-de-de-de-de", 50.05, 40.0),
        ("spring-of-2026", 47.9, 38.0),
        ("deep-house-003", 32.0, 40.0),
    ]
    library = _image_library()
    for track_id, start, length in windows:
        assert start + length <= library[track_id].duration, track_id


# ── the loader and the report ───────────────────────────────────────────────
def _write_library(tmp_path, **changes):
    music = tmp_path / "music"
    music.mkdir()
    (music / "bed.mp3").write_bytes(b"not really audio")
    entry = {
        "id": "bed", "file": "bed.mp3", "duration": 60.0, "title": "Bed", "artist": "Someone",
        "licence": "CC0-1.0", "attribution": 'Music: "Bed" by Someone, CC0 1.0.', **changes,
    }
    entry = {k: v for k, v in entry.items() if v is not None}
    (music / "manifest.json").write_text(json.dumps({"tracks": [entry]}))
    return str(music)


def test_the_service_refuses_a_track_it_could_not_credit(tmp_path):
    with pytest.raises(MusicLibraryError, match="licence must be one of"):
        load_library(_write_library(tmp_path, licence="unknown"))


def test_the_service_refuses_a_track_without_attribution(tmp_path):
    with pytest.raises(MusicLibraryError, match="attribution is required"):
        load_library(_write_library(tmp_path, attribution=None))


def test_a_cc0_track_asks_for_no_credit(tmp_path):
    track = load_library(_write_library(tmp_path))["bed"]
    assert not track.credit_required and track.report()["licence"] == "CC0 1.0"


def test_the_music_fixture_is_the_script_composition_over_deep_house_003():
    bundle = music_bundle()
    assert BUNDLES["music"] is music_bundle and bundle["composition"]["html"] == SCRIPT_COMPOSITION.read_text()
    assert bundle["audio"]["music"] == {"track": "deep-house-003", "start": 32.0}
    assert all("text" in line for line in bundle["audio"]["voice"]["lines"])


def test_a_render_with_music_reports_the_track_its_window_and_its_credit_line(settings):
    parsed = parse_bundle(music_bundle(), settings, _image_library())
    report = music_report(parsed)
    assert report == {
        "track": "deep-house-003",
        "title": "Deep House 003",
        "artist": "Sascha Ende",
        "licence": "CC BY 4.0",
        "licence_url": LICENCES["CC-BY-4.0"]["url"],
        "attribution": 'Music: "Deep House 003" by Sascha Ende (ende.app), licensed CC BY 4.0.',
        "credit_required": True,
        "start": 32.0,
        "end": 41.0,
    }
    silent = parse_bundle({**music_bundle(), "audio": {"voice": music_bundle()["audio"]["voice"]}}, settings, {})
    assert music_report(silent) is None
