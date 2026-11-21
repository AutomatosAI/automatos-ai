/**
 * PRD-277 US-505, PRD-279 US-402 — the enterprise web server stops at start-up when a setting it
 * reads at run time is missing or bad (start-up.ts), with one line naming the setting and never
 * its value, instead of serving pages with a wrong policy or an unreachable API.
 *
 * Node only: instrumentation.ts imports it inside its `NEXT_RUNTIME === 'nodejs'` branch, the
 * shape Next's instrumentation guide gives for runtime-specific code, so `process.exit` never
 * reaches the Edge bundle.
 */
import { enterpriseStartUpProblem, type StartUpSettings } from './start-up'

export const START_UP_REFUSED_EXIT_CODE = 1

type Exit = (code: number) => void

const exitProcess: Exit = (code) => process.exit(code)

/** Stops the server through `exit` when `settings` cannot run the enterprise web app; otherwise nothing. */
export function stopUnlessEnterpriseReady(settings: StartUpSettings = process.env, exit: Exit = exitProcess): void {
  const problem = enterpriseStartUpProblem(settings)
  if (problem === null) return
  console.error(`[start-up] The web app cannot start: ${problem}`)
  exit(START_UP_REFUSED_EXIT_CODE)
}
