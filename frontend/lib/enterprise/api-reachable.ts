/**
 * PRD-277 US-505 — "does the API answer?" for the connection indicator (hooks/use-backend-health.ts)
 * in the enterprise edition.
 *
 * On one host the API's own liveness routes (`/health`, `/health/ready`) are not exposed outside
 * the cluster, by design, so a browser asking `/health` reaches the web app instead and would read
 * every install's API as down. The indicator asks the discovery document instead:
 * `/.well-known/automatos.json` is routed to the API on that host, is public (it needs no sign-in,
 * so asking it never opens the session-ended dialog), reads config only, and is small. Chosen over
 * `/api/system/health`, which is routed too but needs a signed-in caller and does more work.
 *
 * The question is only whether the API answers, so any HTTP answer counts as reachable, a 404 or a
 * 500 included; only a network failure, a timeout, or a gateway's answer that the API behind it is
 * not there (502, 503, 504) counts as not. The body is not read: the indicator's readers in this
 * edition use the connection only (Settings → Session mode, the one reader of /health's body, is
 * shown in the local edition only), so the answer is `null`, never a made-up /health body.
 */

/** The public, routed path the indicator asks in this edition. */
export const API_REACHABLE_PATH = '/.well-known/automatos.json'

/** A gateway's answers that the API behind it is not there. */
const GATEWAY_FAILURES: ReadonlySet<number> = new Set([502, 503, 504])

/** Resolves (to null) when the API answers within `timeoutMs`; rejects when it does not. */
export async function fetchApiReachable(timeoutMs: number): Promise<null> {
  const response = await fetch(API_REACHABLE_PATH, {
    method: 'GET',
    cache: 'no-store',
    credentials: 'omit',
    signal: AbortSignal.timeout(timeoutMs),
  })
  if (GATEWAY_FAILURES.has(response.status)) throw new Error(`The API is not reachable (gateway answered ${response.status}).`)
  return null
}
