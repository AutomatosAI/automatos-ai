"""
Voice latency A-4: the quick-ask runner keeps what it measured
==============================================================

A run's asks cost real model calls, so a failure after they ran must not lose them:

  * the read of the server's split fails (no database within reach): the run is
    reported with ``server_marks=unavailable``;
  * the token file cannot be read part-way: the run stops there and reports the asks
    already timed; before the first ask it is refused; before a speech request that
    ask keeps its chat times.

The server double and the clock are the runner tests' own.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

_orchestrator_root = Path(__file__).resolve().parent.parent
if str(_orchestrator_root) not in sys.path:
    sys.path.insert(0, str(_orchestrator_root))

from tests import test_voice_latency_a4_quick_ask_runner as runner_tests  # noqa: E402

_REPLY_LINES = runner_tests._REPLY_LINES
_TOKEN = runner_tests._TOKEN
_Server = runner_tests._Server
_args = runner_tests._args
_asks = runner_tests._asks
_chat = runner_tests._chat
_no_database = runner_tests._no_database
runner = runner_tests.runner  # the fixture: the CLI module on a ticking clock, with no pause


def test_a_failed_read_after_the_run_keeps_the_runs_own_measures(runner, monkeypatch, caplog):
    def unreachable(workspace_id, chat_ids):
        raise RuntimeError("could not connect to db-host-not-for-logs")

    monkeypatch.setattr(runner, "read_timings", unreachable)
    server = _Server(["spoken_mode", "tts"], [_chat(*_REPLY_LINES)])

    with caplog.at_level(logging.WARNING):
        outcome = runner.quick_ask(_args(runner), _asks(), server)

    assert outcome["server_marks"] == "unavailable" and outcome["server_traces"] == 0
    assert outcome["results_line"].endswith(" · server_marks=unavailable")
    assert "srv_" not in outcome["results_line"]
    turn = outcome["turns"][0]
    assert turn["first_sentence_ms"] is not None and turn["go_first_audio_ms"] is not None
    assert all(turn[name] is None for name in turn if name.startswith("srv_"))
    assert outcome["asks_not_sent"] == 0
    assert "quick_ask_server_marks_unreadable err=RuntimeError" in caplog.text
    assert "db-host-not-for-logs" not in caplog.text


def _token_that_fails_on(runner, monkeypatch, failing_read):
    """A token reader whose ``failing_read``-th read (from 1) finds the file empty."""
    from scripts.eval.voice_latency.run import RunRefused

    reads: list = []

    def reader(path):
        reads.append(path)
        if len(reads) == failing_read:
            raise RunRefused("the token file is empty")
        return _TOKEN

    monkeypatch.setattr(runner, "read_token", reader)
    return reads


def test_a_token_lost_part_way_stops_the_run_and_reports_the_asks_already_timed(runner, monkeypatch, tmp_path):
    _no_database(runner, monkeypatch)
    _token_that_fails_on(runner, monkeypatch, 3)  # ask 1's chat, ask 1's speech, then ask 2's chat
    server = _Server(["spoken_mode", "tts"], [_chat(*_REPLY_LINES), _chat(*_REPLY_LINES), _chat(*_REPLY_LINES)])

    outcome = runner.quick_ask(_args(runner, "--token-file", str(tmp_path / "token")), _asks(3), server)

    assert len(outcome["turns"]) == 1 and outcome["asks_not_sent"] == 2
    assert len(server.requests) == 2  # nothing is sent once the token is gone
    turn = outcome["turns"][0]
    assert turn["first_sentence_ms"] is not None and turn["go_first_audio_ms"] is not None
    assert outcome["results_line"].endswith(" · server_marks=unavailable")


def test_a_token_missing_before_the_first_ask_refuses_the_run(runner, monkeypatch, tmp_path):
    from scripts.eval.voice_latency.run import RunRefused

    _no_database(runner, monkeypatch)
    _token_that_fails_on(runner, monkeypatch, 1)
    server = _Server(["spoken_mode", "tts"], [_chat(*_REPLY_LINES)])

    with pytest.raises(RunRefused, match="token file"):
        runner.quick_ask(_args(runner, "--token-file", str(tmp_path / "token")), _asks(), server)

    assert server.requests == []


def test_a_token_lost_before_the_speech_request_keeps_the_replys_times(runner, monkeypatch, tmp_path):
    _no_database(runner, monkeypatch)
    _token_that_fails_on(runner, monkeypatch, 2)  # ask 1's speech
    server = _Server(["spoken_mode", "tts"], [_chat(*_REPLY_LINES), _chat(*_REPLY_LINES)])

    outcome = runner.quick_ask(_args(runner, "--token-file", str(tmp_path / "token")), _asks(2), server)

    assert len(outcome["turns"]) == 2 and outcome["asks_not_sent"] == 0
    first, second = outcome["turns"]
    assert first["first_sentence_ms"] is not None and first["full_ms"] is not None
    assert first["tts_first_byte_ms"] is None and first["go_first_audio_ms"] is None
    assert second["tts_first_byte_ms"] is not None and second["go_first_audio_ms"] is not None
    assert len(server.sent_to("/api/voice/speech")) == 1
