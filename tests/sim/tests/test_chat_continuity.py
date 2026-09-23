"""F079 — a conversation continues under the id the backend persisted it under.

Night 3: every persona continuation became a fresh chat (98 chats, all single
turn). ``api/chat.py`` creates a NEW chat for an id it does not know — the one
the client minted — and names the real one in the stream's first frame; the
parser kept the minted id, so the next turn sent it again and started over.
And a failed turn's ``e:{"message", "code"}`` was read as an SDK finish step,
so ``chat`` printed no error at all.

The frames are the ones ``consumers/chatbot/streaming.py`` emits.
"""

from tests.sim import driver_chat, sse
from tests.sim.api import Response, Trace
from tests.sim.config import Settings
from tests.sim.packs import Scenario
from tests.sim.results import RunContext

CHAT_ID = 'd:{"type":"chat-id","chatId":"7c9e6679-7425-40de-944b-e07fc1f90ae7"}'
ACTIVATION_FAILED = ('e:{"message": "NEWSROOM could not be started \\u2014 it is a session agent, its work runs '
                     'as a board ticket.", "code": "activation_failed"}')


def test_the_backend_chat_id_wins_over_the_id_the_client_sent():
    turn = sse.parse_data_stream(CHAT_ID + '\n0:"hello"', header_chat_id="client-minted-id")
    assert turn.chat_id == "7c9e6679-7425-40de-944b-e07fc1f90ae7"


def test_a_platform_turn_error_is_captured_with_its_code():
    turn = sse.parse_data_stream(CHAT_ID + "\n" + ACTIVATION_FAILED, header_chat_id="client-minted-id")
    assert turn.errors == ("activation_failed: NEWSROOM could not be started — it is a session agent, "
                           "its work runs as a board ticket.",)
    assert turn.chat_id == "7c9e6679-7425-40de-944b-e07fc1f90ae7"


def test_an_error_without_a_code_keeps_its_message():
    assert sse.parse_data_stream('e:{"message": "Turn failed"}').errors == ("Turn failed",)


def test_an_sdk_finish_step_is_not_an_error():
    turn = sse.parse_data_stream('e:{"finishReason":"stop","usage":{"promptTokens":3},"isContinued":false}')
    assert turn.errors == ()


def test_without_a_chat_id_frame_the_sent_id_still_stands():
    assert sse.parse_data_stream('0:"hi"', header_chat_id="sent").chat_id == "sent"


class _ChatApi:
    """Answers each /api/chat with the backend's id, records what was sent."""

    def __init__(self):
        self.sent = []

    def request(self, method, path, *, json_body=None, **_kw):
        self.sent.append(json_body)
        body = CHAT_ID + f'\n0:"reply {len(self.sent)}"'
        return Response(200, body, 5, {}, 1)


def test_a_scripted_chat_continues_under_the_backend_id(monkeypatch):
    monkeypatch.setattr(driver_chat, "snapshot_effects", lambda api, label: {"tasks": frozenset(), "agents": frozenset()})
    monkeypatch.setattr(driver_chat, "answer_pending", lambda api, persona, label: [])
    api = _ChatApi()
    ctx = RunContext(settings=Settings(), api=api, trace=Trace(), workspace_id="ws", agents={}, persona={},
                     run_started="")
    result = driver_chat.run_chat(ctx, Scenario(id="s1", kind="chat", turns=("first", "second")))
    backend = "7c9e6679-7425-40de-944b-e07fc1f90ae7"
    assert api.sent[1]["id"] == api.sent[1]["chatId"] == backend      # turn 2 continues, not a new chat
    assert result.artifacts["chat_id"] == backend
