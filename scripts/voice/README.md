# Auto Live on the local stack

Retell's servers have to reach two backend endpoints: the custom-LLM socket
(`wss://<host>/api/voice/retell/llm-websocket`) and the call-events webhook
(`https://<host>/api/voice/retell/events`). On Railway, `<host>` is the
service's public domain. On the local stack it is an ngrok static domain
that tunnels to the backend (PRD-258 US-003, decision D3).

The backend builds both URLs from `PUBLIC_API_HOST` when Auto Live is armed.
A first arm creates the agent. On the local edition the agent is named
**Auto Live (local)**, so it can't be confused with prod's agent in a shared
Retell dashboard. A re-arm moves an existing agent to the current
`PUBLIC_API_HOST`, so arming again after a host change is enough.

## Only Retell's routes go through the tunnel

On the local edition, an anonymous request is treated as the instance's super
admin (`orchestrator/core/auth/hybrid.py`). A plain tunnel would therefore
publish the whole API, admin routes included, to anyone who finds the
domain. `tunnel.sh` passes `ngrok-policy.yml`, which refuses (404) every path
outside `/api/voice/retell/`. Those two routes carry their own credentials:
the socket needs a call id minted by the backend, and the webhook needs
Retell's HMAC signature. Don't run `ngrok http` against the backend without
that policy.

## Steps, in order

1. **Claim an ngrok static domain.** One free ngrok account gives you one
   static domain, so the host stays the same between runs and needs no DNS.
   Install ngrok and run `ngrok config add-authtoken <token>` once.
2. **Set `PUBLIC_API_HOST` in the local compose override**, never in `.env`.
   Add it to the backend's `environment:` in `docker-compose.override.yml`
   at the repo root (Compose loads that file automatically). The override is
   personal to your machine, so don't commit it.

   ```yaml
   services:
     backend:
       environment:
         PUBLIC_API_HOST: <your static domain>
   ```

   Use the bare host: no `https://` and no trailing slash.
3. **Restart the backend** so it reads the new host:
   `docker compose up -d backend`.
4. **Run the tunnel** and leave it running:

   ```bash
   NGROK_DOMAIN=<your static domain> scripts/voice/tunnel.sh
   ```

   The script checks that ngrok is installed and `NGROK_DOMAIN` is set, then
   prints the `PUBLIC_API_HOST` line to use. It tunnels to the backend's
   published port, which it reads from the `backend` port mapping in
   `docker-compose.yml` (`${API_PORT:-8000}:8000`). As with Compose,
   `API_PORT` in the shell or in `.env` overrides that port.
5. **Arm Auto Live:** Settings → Voice → Auto Live. Paste the Retell API key
   and click Arm. In the Retell dashboard, the agent's custom-LLM URL should
   now be `wss://<your static domain>/api/voice/retell/llm-websocket`.

Don't commit your domain, the ngrok authtoken or a Retell key anywhere.
