/**
 * PRD-261 C-03b — the words the admin theme review queue shows, and the reading of a scan report.
 *
 * The scan report is `theme_packages.scan_report` as the reviewer gets it from
 * `GET /api/admin/themes/{packageId}/report` (`services/themes/review_stage.py`, schema
 * `automatos-theme-scan/v2`): every check carries a `level` a reviewer reads as Error, Warning or
 * Pass, plus `not_measured`. A row stored before W-C1 (schema v1, `services/themes/scan.py`) has a
 * `status` instead (`pass`, `absent`, `not_run`); it is shown by that status, never as passed.
 *
 * The report is server data read defensively: an unknown check is listed by its key, an unknown
 * level by its value.
 */

export const QUEUE_EMPTY = 'No theme versions are waiting for review.'
export const ADMIN_ONLY = 'Only an Automatos admin can review Marketplace themes.'
export const PICK_A_VERSION = 'Choose a version on the left to read its report.'
export const LLM_DID_NOT_RUN = 'LLM stage did not run'
export const LOCAL_REVIEW_NOTE =
  'Local edition: there is no isolated review job here, so your approval is the review.'
export const NOTE_NEEDS_DIGEST = 'A passing note must hold the kit summary line'
/** api/admin_themes.py REASON_MAX: the longest reject or deactivate reason the server takes. */
export const REASON_MAX = 2000
/** A stored signature an active Automatos signing key verifies, when the report names no key. */
export const SIGNATURE_VERIFIED = 'Verified with an active Automatos key.'
/** A stored signature no active Automatos signing key verifies (signature_verify.py `invalid`). */
export const SIGNATURE_INVALID = 'The signature does not verify against an active Automatos key.'

/**
 * A stored signature in words, by the report's check on it (review.py report: `signatureVerified`
 * and `keyId`, from signature_verify.package_verdict, C-05b, read through the hosted feed's revoked
 * keys in the local edition). Verified names the active key that verifies it ("Verified · <keyId>",
 * C-05j). Invalid says no active key does, and names the retired or revoked key that does verify it
 * when there is one (signature_verify.verify).
 */
export function storedSignatureWords(verified: boolean, keyId: string | null): string {
  if (verified) return keyId ? `Verified · ${keyId}` : SIGNATURE_VERIFIED
  return keyId ? `${SIGNATURE_INVALID} Only ${keyId}, a retired or revoked key, verifies it.` : SIGNATURE_INVALID
}

// ===== PRD-261 C-05j: the review job's inputs (C-03c) and the revocations panel (C-05e, C-05h) =====
export const REVIEW_JOB_INPUTS = 'Review job inputs'
export const GET_REVIEW_JOB_INPUTS = 'Get the review job inputs'
/** review_package.py REVIEW_LINK_SECONDS: how long a review link works. */
export const REVIEW_LINK_MINUTES = 15
export const REVIEW_JOB_INPUTS_HINT =
  `Give the kit's Theme review workflow these two inputs. The link is made fresh each time and works for ${REVIEW_LINK_MINUTES} minutes.`

/** When a review link stops working, as a time of day in the reader's own clock. */
export function linkExpiryWords(expiresAt: string): string {
  const at = new Date(expiresAt)
  if (Number.isNaN(at.getTime())) return `The link stops working ${REVIEW_LINK_MINUTES} minutes after it was made.`
  return `The link stops working at ${at.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}.`
}

export const REVOCATIONS_TITLE = 'Theme revocations'
export const REVOCATIONS_LEDE =
  'A revocation names a package digest, or a theme id with an optional version range. Warn keeps the theme running with a notice; disable shows members its base instead, on their next load.'
export const NO_REVOCATIONS = 'No theme has been revoked.'
export const REVOKE = 'Revoke'
export const LIFT = 'Lift'
export const CONFIRM_LIFT = 'Confirm lift'
export const KEEP_IT = 'Keep it'
export const REVOCATION_CREATED = 'Revocation saved. Members see it on their next load.'
export const REVOCATION_REPEATED = 'A live revocation already says this, so nothing new was written.'
export const REVOCATION_LIFTED = 'Revocation lifted. Members get the theme back on their next load.'
/** revocation_admin.py REASON_MAX: the longest reason the server takes. */
export const REVOCATION_REASON_MAX = 2000

export const LEVEL_WORDS: Readonly<Record<string, string>> = {
  warn: 'Warn',
  disable: 'Disable',
}

export const LEVEL_HINTS: Readonly<Record<string, string>> = {
  warn: 'Warn: the theme keeps running, with a notice',
  disable: 'Disable: members see its base instead',
}

/** theme_revocations.origin: written in this install, or pulled from the hosted feed (C-05h). */
export const ORIGIN_WORDS: Readonly<Record<string, string>> = {
  local: 'Set here',
  feed: 'From the Automatos feed',
}

export const SYNC_TITLE = 'Revocation sync'
export const CHECK_NOW = 'Check now'
export const SYNC_OFF =
  "Revocation sync is off: this install has no feed address (THEME_REVOCATIONS_URL), so it doesn't follow the Automatos revocation list."
export const SYNC_STALE =
  "No check of the Automatos revocation list has worked in the last 7 days. Until one does, this install may miss a revocation, and feed rows can be lifted here."
export const SYNC_CHECKED = 'Revocation list checked.'

/** A revocation's target in words: its digest, or its theme id with the range (every version without one). */
export function revocationTargetWords(row: {
  readonly digest: string | null
  readonly themeId: string | null
  readonly versionRange: string | null
}): string {
  if (row.digest) return `Package ${row.digest}`
  return `${row.themeId ?? 'Unknown theme'} ${row.versionRange ?? 'every version'}`
}

/** The kit's summary line naming the version (`services/themes/review.py` NOTE_DIGEST_LINE). */
export function noteDigestLine(digest: string): string {
  return `Package digest:  ${digest}`
}

/**
 * True when the pasted summary holds the line naming this digest. A hint for the form only: the
 * server checks the kit's header block and its one `Outcome: pass` line (`note_names_this_version`).
 */
export function noteNamesDigest(summary: string, digest: string): boolean {
  const wanted = noteDigestLine(digest)
  return summary.split(/\r?\n/).some((line) => line.trim() === wanted)
}

export type CheckLevel = 'pass' | 'warning' | 'error' | 'not_measured' | 'not_run' | 'absent' | string

export const LEVEL_LABELS: Readonly<Record<string, string>> = {
  pass: 'Pass',
  warning: 'Warning',
  error: 'Error',
  not_measured: 'Not measured',
  not_run: 'Not run',
  absent: 'Absent',
}

export const VERDICT_LABELS: Readonly<Record<string, string>> = {
  pass: 'Pass',
  review_required: 'Review required',
  blocked: 'Blocked',
  not_run: 'Not run',
}

export const STATUS_LABELS: Readonly<Record<string, string>> = {
  pending: 'Pending',
  approved: 'Approved',
  rejected: 'Rejected',
  deactivated: 'Deactivated',
}

export const CHECK_LABELS: Readonly<Record<string, string>> = {
  sizes: 'Sizes',
  cssScope: 'Style sheet scope',
  looks: 'Looks',
  contrast: 'Contrast (AA)',
  uses: 'SDK names used',
  audioCapture: 'Microphone and speech',
  clonedFrom: 'Cloned pages',
  llm: 'LLM review',
  automated: 'Automated review',
}

export function labelOf(labels: Readonly<Record<string, string>>, value: string | null | undefined): string {
  if (!value) return labels.not_run ?? 'Unknown'
  return labels[value] ?? value
}

/** A finding of the LLM stage (`llm_stage._finding`). */
export interface ReportFinding {
  readonly severity: string
  readonly category: string
  readonly description: string
  readonly file: string | null
  readonly evidence: string | null
}

/** One check of the report, as the page shows it. */
export interface ReportCheck {
  readonly key: string
  readonly level: CheckLevel
  readonly reason: string | null
  readonly problems: readonly string[]
  /** The LLM check only: true when it did not read all of the code (`status` not `ran`). */
  readonly llmDidNotRun: boolean
  readonly riskScore: number | null
  readonly summary: string | null
  readonly findings: readonly ReportFinding[]
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function text(value: unknown): string | null {
  return typeof value === 'string' && value ? value : null
}

function strings(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : []
}

function findingOf(value: unknown): ReportFinding | null {
  if (!isRecord(value)) return null
  return {
    severity: text(value.severity) ?? 'unknown',
    category: text(value.category) ?? 'finding',
    description: text(value.description) ?? '',
    file: text(value.file),
    evidence: text(value.evidence),
  }
}

function llmDidNotRun(key: string, check: Record<string, unknown>): boolean {
  if (key !== 'llm') return false
  return check.status !== 'ran'
}

function checkOf(key: string, value: unknown): ReportCheck {
  const check = isRecord(value) ? value : {}
  const findings = Array.isArray(check.findings) ? check.findings.map(findingOf) : []
  return {
    key,
    level: text(check.level) ?? text(check.status) ?? 'unknown',
    reason: text(check.reason),
    problems: strings(check.problems),
    llmDidNotRun: llmDidNotRun(key, check),
    riskScore: typeof check.riskScore === 'number' ? check.riskScore : null,
    summary: text(check.summary),
    findings: findings.filter((item): item is ReportFinding => item !== null),
  }
}

/** The report's checks in the server's order; an empty list for a report this page can't read. */
export function reportChecksOf(report: unknown): ReportCheck[] {
  const checks = isRecord(report) && isRecord(report.checks) ? report.checks : {}
  return Object.entries(checks).map(([key, value]) => checkOf(key, value))
}

export const SCAN_BLOCKED = 'The automated review blocked this version: reject it with the reason.'
export const SCAN_NOT_RUN = "The automated review hasn't run on this version."

/**
 * Why approve is disabled whatever the server's `approval` says: a blocked or not-run scan never
 * passes (`review.approval_refusal`, unlike `admin_plugins.py:222-223`).
 */
export function scanRefusal(scanVerdict: string | null): string | null {
  if (scanVerdict === 'blocked') return SCAN_BLOCKED
  if (scanVerdict === 'not_run') return SCAN_NOT_RUN
  return null
}

/** Why approve is disabled: the scan first, then the server's own reason; null when it may be approved. */
export function approveRefusal(
  scanVerdict: string | null,
  approval: { readonly allowed: boolean; readonly reason: string | null },
): string | null {
  const scan = scanRefusal(scanVerdict)
  if (scan) return scan
  if (approval.allowed) return null
  return approval.reason ?? 'The server says this version cannot be approved yet.'
}
