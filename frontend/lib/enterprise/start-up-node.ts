/**
 * PRD-277 US-505, PRD-279 US-402 — the enterprise web server stops at start-up when a setting it
 * reads at run time is missing or bad (start-up.ts), with one line naming the setting and never
 * its value, instead of serving pages with a wrong policy or an unreachable API.
 *
 * Never during `next build`: the run-time settings are not there yet, and stopping would stop the
 * image being built. Next names that phase `phase-production-build` (`PHASE_PRODUCTION_BUILD`,
 * next/constants) and sets it in `process.env.NEXT_PHASE` before its build workers start; Next 15
 * itself skips register() in that phase, and this check skips it again in case a build ever calls it.
 *
 * Node only: instrumentation.ts imports it inside its `NEXT_RUNTIME === 'nodejs'` branch, the
 * shape Next's instrumentation guide gives for runtime-specific code, so `process.exit` never
 * reaches the Edge bundle.
 */
import { enterpriseStartUpProblem, type StartUpSettings } from './start-up'

export const START_UP_REFUSED_EXIT_CODE = 1

/** Next's build phase (next/constants PHASE_PRODUCTION_BUILD), as `process.env.NEXT_PHASE` holds it. */
export const BUILD_PHASE = 'phase-production-build'

/** What the check reads: the settings, and the phase Next runs in. */
export interface StartUpEnvironment extends StartUpSettings {
  readonly NEXT_PHASE?: string
}

type Exit = (code: number) => void

const exitProcess: Exit = (code) => process.exit(code)

/** Stops the server through `exit` when `env` cannot run the enterprise web app; otherwise, and in a build, nothing. */
export function stopUnlessEnterpriseReady(env: StartUpEnvironment = process.env, exit: Exit = exitProcess): void {
  if (env.NEXT_PHASE === BUILD_PHASE) return
  const problem = enterpriseStartUpProblem(env)
  if (problem === null) return
  console.error(`[start-up] The web app cannot start: ${problem}`)
  exit(START_UP_REFUSED_EXIT_CODE)
}
