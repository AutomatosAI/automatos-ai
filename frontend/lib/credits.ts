/**
 * PRD-265 — the credit stop codes a chat turn can end with, mirrored from the
 * backend so both ends reference ONE string.
 *
 * Backend source of truth: `orchestrator/services/credits/routing.py`
 * (`CREDITS_EXHAUSTED_CODE`). 1 credit = 1 US cent of AI usage at Automatos
 * prices; every platform-funded call is metered at its real cost.
 */

/**
 * Platform-funded work is paused: the workspace has no credits left for work on
 * Automatos's AI keys (or a run's cost reached what was left). Calls on the
 * workspace's own keys keep working. The chat renders the paused banner on it
 * (TrialExhaustedBanner).
 */
export const CREDITS_EXHAUSTED_CODE = 'credits_exhausted'
