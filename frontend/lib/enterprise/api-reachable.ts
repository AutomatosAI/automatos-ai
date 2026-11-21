/**
 * PRD-277 US-505 — "does the API answer?" for the connection indicator (hooks/use-backend-health.ts)
 * in the enterprise edition.
 *
 * On one host the API's own liveness routes (`/health`, `/health/ready`) are not exposed outside
 * the cluster, by design, so a browser asking `/health` reaches the web app instead and would read
 * every install's API as down. The indicator asks the sign-in page's route instead:
 * `/api/v1/auth/sign-in-page` (api/auth_web.py) is under `/api`, so it reaches the API on every
 * install with no extra route, is public (it needs no sign-in, so asking it never opens the
 * session-ended dialog), reads one setting, does no database work, and answers 200. It is not one
 * of the two sign-in steps limited per address. Chosen over the discovery document, which answers
 * an error in this edition until phone sign-in for it is built (US-602), and over
 * `/api/system/health`, which needs a signed-in caller and does more work.
 *
 * The question is only whether the API answers, so any HTTP answer counts as reachable, a 404 or a
 * 500 included; only a network failure, a timeout, or a gateway's answer that the API behind it is
 * not there (502, 503, 504) counts as not. The body is not read: the indicator's readers in this
 * edition use the connection only (Settings → Session mode, the one reader of /health's body, is
 * shown in the local edition only), so the answer is `null`, never a made-up /health body.
 */

/** The public path the indicator asks in this edition (lib/enterprise/auth-api.ts's SIGN_IN_PAGE_PATH). */
export const API_REACHABLE_PATH = '/api/v1/auth/sign-in-page'

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
