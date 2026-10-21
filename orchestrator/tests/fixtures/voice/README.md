# Retell create-web-call responses (PRD-266 US-008a)

Response bodies for `test_prd266_voice_mint.py`, written from Retell's API reference
(21 Oct 2026), not captured from a live call. The tokens, ids and TURN credentials are
short placeholders.

- `retell_create_web_call_transport.json`: the connection-details body Retell documents
  for `/v3/create-web-call` (`call_id`, `access_token`, `transport`, `ice_servers`,
  `expires_at`), plus a `url`. Retell's reference does not list `url`; it is here to show
  that the mint passes it through when Retell sends it.
- `retell_create_web_call_call_object.json`: the call object that `/v2/create-web-call`
  returns (the endpoint the mint uses). It has none of the transport fields, so the mint
  returns only `call_id`, `access_token` and `chat_id`.
