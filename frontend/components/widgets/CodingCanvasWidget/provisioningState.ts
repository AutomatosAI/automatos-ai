/**
 * Canvas provisioning state machine
 * PRD-170 S2
 *
 * Wizard-created workspaces have no directory on the (shared, sandboxed) worker
 * volume until first canvas open. This models the honest UI lifecycle of opening
 * the canvas — provisioning progress, ready, and a failed state with retry — so
 * the panel never shows a blank/broken surface while the workspace is being
 * provisioned.
 *
 * DECISION (v1): the canvas reuses the EXISTING shared sandboxed worker runner
 * (per-workspace directory + server-side confinement — see canvas_confinement),
 * NOT a per-workspace container. Per-workspace containers are the isolation
 * upgrade path, but the deployed architecture is a single worker service keyed
 * by workspace_id (core/workspace_client._worker_url), and confinement already
 * fences each session to its mount. This memo is why S2 provisions a directory,
 * not a container.
 *
 * Pure + immutable (no React, no I/O) so it is unit-testable; the panel drives
 * it from the start-session request outcome.
 */

export type ProvisioningPhase =
  | 'idle'
  | 'provisioning'
  | 'ready'
  | 'failed'

export interface ProvisioningState {
  phase: ProvisioningPhase
  /** True when this open COLD-provisioned the workspace (fresh directory). */
  coldProvisioned: boolean
  error: string | null
  /** Retry count — the UI shows it, and can cap retries. */
  attempts: number
}

export const initialProvisioningState: ProvisioningState = {
  phase: 'idle',
  coldProvisioned: false,
  error: null,
  attempts: 0,
}

/** User (or effect) initiated a canvas open — enter provisioning. */
export function beginProvisioning(state: ProvisioningState): ProvisioningState {
  return {
    ...state,
    phase: 'provisioning',
    error: null,
    attempts: state.attempts + 1,
  }
}

export interface StartSessionOutcome {
  success: boolean
  /** The worker reports whether THIS open created the workspace directory. */
  provisioned?: boolean
  error?: string
}

/** Fold the start-session request outcome into the provisioning state. */
export function resolveProvisioning(
  state: ProvisioningState,
  outcome: StartSessionOutcome
): ProvisioningState {
  if (outcome.success) {
    return {
      ...state,
      phase: 'ready',
      coldProvisioned: Boolean(outcome.provisioned),
      error: null,
    }
  }
  return {
    ...state,
    phase: 'failed',
    error: outcome.error || 'Failed to start the canvas session',
  }
}

/** True when the UI should offer a Retry affordance. */
export function canRetry(state: ProvisioningState, maxAttempts = 3): boolean {
  return state.phase === 'failed' && state.attempts < maxAttempts
}

/** Human-readable status line for the provisioning banner. */
export function provisioningLabel(state: ProvisioningState): string {
  switch (state.phase) {
    case 'provisioning':
      return state.attempts > 1
        ? `Retrying (attempt ${state.attempts})…`
        : 'Provisioning workspace…'
    case 'ready':
      return state.coldProvisioned ? 'Workspace provisioned' : 'Session ready'
    case 'failed':
      return state.error ?? 'Provisioning failed'
    case 'idle':
    default:
      return ''
  }
}
