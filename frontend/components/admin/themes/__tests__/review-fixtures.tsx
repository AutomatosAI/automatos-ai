/**
 * PRD-261 C-03b — rows for the admin theme review queue tests, in the wire shapes of
 * lib/api-client.ts (the answers of orchestrator/api/admin_themes.py through services/themes/review.py).
 */
import { render } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { ReactElement } from 'react'

import type {
  ThemePendingVersion,
  ThemeReviewPackageLink,
  ThemeReviewReport,
  ThemeRevocationRow,
  ThemeRevocationSyncStatus,
  ThemeSignatureRequest,
} from '@/lib/api-client'

export const PACKAGE_ID = '44444444-4444-4444-8444-444444444444'
export const REQUEST_ID = '55555555-5555-4555-8555-555555555555'
export const DIGEST = 'b'.repeat(64)
/** signature_verify.py key_id_of: `ed25519:` and 16 hex. An active key, and a retired one. */
export const KEY_ID = 'ed25519:0123456789abcdef'
export const RETIRED_KEY_ID = 'ed25519:fedcba9876543210'

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
    keyId: null,
    ...overrides,
  }
}

export const REVOCATION_ID = '77777777-7777-4777-8777-777777777777'
export const FEED_REVOCATION_ID = '88888888-8888-4888-8888-888888888888'
const DAY_MS = 24 * 60 * 60 * 1000

/** An ISO time `days` days before now, with its offset (formatRelative's input). */
export function daysAgo(days: number): string {
  return new Date(Date.now() - days * DAY_MS).toISOString()
}

/** revocation_admin.py revocation_view: a live local disable of the fixture's digest. */
export function revocationRow(overrides: Partial<ThemeRevocationRow> = {}): ThemeRevocationRow {
  return {
    id: REVOCATION_ID,
    digest: DIGEST,
    themeId: null,
    versionRange: null,
    level: 'disable',
    reason: 'Sends chat text to another site.',
    origin: 'local',
    createdAt: daysAgo(1),
    liftedAt: null,
    ...overrides,
  }
}

/** revocation_pull.py sync_status: sync on, a pull that worked an hour ago. */
export function syncStatus(overrides: Partial<ThemeRevocationSyncStatus> = {}): ThemeRevocationSyncStatus {
  const anHourAgo = new Date(Date.now() - DAY_MS / 24).toISOString()
  return {
    enabled: true,
    lastAttemptAt: anHourAgo,
    lastSuccessAt: anHourAgo,
    lastError: null,
    feedGeneratedAt: anHourAgo,
    rows: 1,
    revokedKeys: [],
    stale: false,
    ...overrides,
  }
}

/** review_package.py review_link: the review job's two inputs and when the link stops working. */
export function reviewLink(overrides: Partial<ThemeReviewPackageLink> = {}): ThemeReviewPackageLink {
  return {
    url: `https://storage.example/themes/_review/${DIGEST}.autotheme?expires=900`,
    digest: DIGEST,
    expiresAt: new Date(Date.now() + DAY_MS / 96).toISOString(),
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
