/**
 * PRD-261 C-03b — rows for the admin theme review queue tests, in the wire shapes of
 * lib/api-client.ts (the answers of orchestrator/api/admin_themes.py through services/themes/review.py).
 */
import { render } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { ReactElement } from 'react'

import type { ThemePendingVersion, ThemeReviewReport, ThemeSignatureRequest } from '@/lib/api-client'

export const PACKAGE_ID = '44444444-4444-4444-8444-444444444444'
export const REQUEST_ID = '55555555-5555-4555-8555-555555555555'
export const DIGEST = 'b'.repeat(64)

// review_stage.py run_review_stage: schema v2, one check per level, and the LLM stage that had no key.
export const SCAN_REPORT_V2 = {
  schema: 'automatos-theme-scan/v2',
  stage: 'C-02a',
  verdict: 'review_required',
  summary: { pass: 3, warning: 1, error: 1, not_measured: 1 },
  checks: {
    sizes: { level: 'pass', packageBytes: 20480 },
    cssScope: { level: 'pass', path: 'dist/theme.css' },
    looks: { level: 'pass', presets: ['default'] },
    contrast: { level: 'not_measured', reason: 'Not measured: no colour pairs.' },
    uses: { level: 'error', status: 'checked', problems: ['useSecret is not an SDK name.'] },
    llm: {
      level: 'warning', status: 'did_not_run', reason: 'LLM stage did not run: no API key is set.',
      riskScore: 0, findings: [], summary: '',
    },
  },
}

// step_up.py step_up_body(10).
export const STEP_UP_BODY = {
  clerk_error: {
    type: 'forbidden',
    reason: 'reverification-error',
    metadata: { reverification: { level: 'first_factor', afterMinutes: 10 } },
  },
  code: 'step_up_required',
  detail: "Confirm it's you to continue: this action needs a sign-in from the last 10 minutes.",
}

export function stepUpError(): Error {
  return Object.assign(new Error(STEP_UP_BODY.detail), { status: 403, body: STEP_UP_BODY })
}

export function httpError(status: number, detail: string): Error {
  return Object.assign(new Error(detail), { status, body: { detail } })
}

export function pendingRow(overrides: Partial<ThemePendingVersion> = {}): ThemePendingVersion {
  return {
    id: PACKAGE_ID,
    themeId: 'automatos-e2e/hello',
    version: '1.0.0',
    name: 'Hello theme',
    publisher: 'automatos-e2e',
    digest: DIGEST,
    status: 'pending',
    reviewed: false,
    scanVerdict: 'review_required',
    submittedBy: 7,
    signed: false,
    createdAt: '2026-10-19T10:00:00+00:00',
    reportNote: null,
    approval: { allowed: true, reason: null },
    ...overrides,
  }
}

export function signatureRequest(): ThemeSignatureRequest {
  return {
    requestId: REQUEST_ID,
    payload: { id: 'automatos-e2e/hello', version: '1.0.0', digest: DIGEST, files: { 'dist/entry.js': 'c'.repeat(64) } },
    payloadSha256: 'd'.repeat(64),
    algorithm: 'ed25519',
    form: 'automatos-theme-sign/v1',
    canonical: '{"digest":"…"}',
  }
}

export function reportOf(overrides: Partial<ThemeReviewReport> = {}): ThemeReviewReport {
  const { reportNote, approval, ...summary } = pendingRow()
  return {
    ...summary,
    reportNote,
    approval,
    manifest: { id: 'automatos-e2e/hello', version: '1.0.0' },
    scanReport: SCAN_REPORT_V2,
    history: [{ event: 'submitted', from: null, to: 'pending', by: 7, reason: null, at: '2026-10-19T10:00:00+00:00' }],
    signatureRequest: null,
    signatureVerified: false,
    ...overrides,
  }
}

/** The kit's summary for this digest (the kit's scripts/review/summary.ts header). */
export function kitSummary(digest: string, outcome: 'pass' | 'fail' = 'pass'): string {
  return ['Theme review: automatos-e2e/hello 1.0.0', `Package digest:  ${digest}`, `Outcome: ${outcome}`, '', 'Checks:'].join('\n')
}

export function renderWithQuery(ui: ReactElement) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>)
}
