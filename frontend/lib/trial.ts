/**
 * PRD-222 US-014 (W1·S9) — trial protocol constants shared by the frontend
 * surfaces (the exhausted banner, the chat error path) and mirrored from the
 * backend so both ends reference ONE string.
 *
 * Backend source of truth: `services/trial_ledger.TRIAL_EXHAUSTED_CODE`. The
 * LLM-dispatch gate raises `TrialExhaustedError(error_code='trial_exhausted')`
 * the instant a trial workspace is blocked; it also flips
 * `workspaces.onboarding.trial.state` to `exhausted` on that same request, so
 * the exhausted state is reachable either from the live error or from the
 * US-002 workspace snapshot.
 */
export const TRIAL_EXHAUSTED_CODE = 'trial_exhausted'
