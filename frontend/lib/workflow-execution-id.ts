/**
 * #1664: a workflow execution's id as it goes into the API's path. The web app's stream proxy
 * (`app/api/workflows/stream/route.ts`) forwards to `/api/workflows/executions/{id}/stream/aisdk`,
 * whose id is an int (`WorkflowExecution.id`), so only a positive whole number passes: it keeps
 * anything else out of the forwarded URL's path (OWASP A10:2021 - SSRF). The browser sends it as a
 * number (`workflow-stream-viewer.tsx`) or, in a query string, as its digits.
 */

const EXECUTION_ID_DIGITS = /^[1-9][0-9]*$/

/** The execution id as the path's digits, or null when it isn't a positive safe integer. */
export function executionIdPath(value: unknown): string | null {
  const text = typeof value === 'number' ? String(value) : value
  if (typeof text !== 'string' || !EXECUTION_ID_DIGITS.test(text)) return null
  return Number.isSafeInteger(Number(text)) ? text : null
}
