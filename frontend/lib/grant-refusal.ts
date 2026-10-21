/**
 * Who may answer, as the server decided it (PRD-266 D16, US-006b).
 *
 * The approval-grants routes answer owners, admins and editors (the backend's
 * `require_workspace_answerer`); a viewer gets 403. The Questions tab and the
 * Governance approvals inbox never work that out themselves: they read the status
 * the api client carries on its error, say the refusal plainly, and keep their
 * own load-error line for any other failure.
 */

const FORBIDDEN_STATUS = 403

/** The line a refused (403) read of the approval grants shows. */
export const ANSWERER_REFUSAL_LINE = 'Only owners, admins and editors can answer questions.'

/** The refusal line for a 403, else the surface's own `loadError`. */
export function grantsLoadErrorLine(error: unknown, loadError: string): string {
  const status = (error as { status?: unknown } | null)?.status
  return status === FORBIDDEN_STATUS ? ANSWERER_REFUSAL_LINE : loadError
}
