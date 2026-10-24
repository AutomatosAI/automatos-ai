"""A message sent with a photo reaches the answer checks as its words (24 Oct 2026).

The iPhone app sent "What you see now" with a photo. The model answered, then the tool loop's
answer checks (F187 ``invented_ids``, F205 ``internal_names``) ran their regexes over the
person's message, which ``_extract_user_text`` returned as the raw content: a list of parts
(text plus ``image_url``). They raised "expected string or bytes-like object, got 'list'" and
the reply failed ("Auto couldn't answer that"). The person's words are now the text parts.
"""
from consumers.chatbot.claim_check import invented_ids
from consumers.chatbot.service import StreamingChatService

PHOTO = {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/"}}
WORDS = "What you see now"


def _user_text(messages):
    # _extract_user_text reads only its argument; no service is needed to call it.
    return StreamingChatService._extract_user_text(None, messages)


def test_a_photo_message_gives_its_words_not_its_parts():
    messages = [
        {"role": "system", "content": "You are Auto."},
        {"role": "user", "content": [{"type": "text", "text": WORDS}, PHOTO]},
    ]
    assert _user_text(messages) == WORDS


def test_a_photo_sent_alone_gives_no_words():
    assert _user_text([{"role": "user", "content": [PHOTO]}]) == ""


def test_a_plain_message_is_unchanged():
    assert _user_text([{"role": "user", "content": "Hi"}]) == "Hi"


def test_the_latest_person_message_wins_over_earlier_photo_turns():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "Can you see this"}, PHOTO]},
        {"role": "assistant", "content": "Yes."},
        {"role": "user", "content": [{"type": "text", "text": WORDS}, PHOTO]},
    ]
    assert _user_text(messages) == WORDS


def test_the_answer_check_takes_the_words_without_raising():
    owner = _user_text([{"role": "user", "content": [{"type": "text", "text": WORDS}, PHOTO]}])
    assert isinstance(owner, str)
    # no id-shaped token in the answer: nothing is looked up, and nothing raises on the owner's text
    assert invented_ids("It's a laptop on a desk.", owner, "00000000-0000-0000-0000-000000000000") == []
