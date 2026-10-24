/**
 * PRD-261 A-07d — the scan report of a private upload (PRD §5), as `POST /api/themes/import` sends it:
 * `review_stage.member_view` of the stored report.
 *
 * Since W-C1 that is the review stage's report (`services/themes/review_stage.py`, schema
 * `automatos-theme-scan/v2`): every check carries a `level` (pass, warning, error, not_measured)
 * beside its own `status`, `reason` and `problems`, and the AI code read (`llm`) is left out. A row
 * stored before it has the early report (`scan.py`, v1): a `status` per check and no level. A check
 * is shown by its level when it has one, else by its status; one that did not run says so with the
 * server's reason, never as passed. When the verdict is worse than every check shown, the one line
 * under it says the AI code read decided it.
 *
 * The report is data from the server, read defensively: an unknown check is listed by its key, an
 * unknown level or status by its value.
 */
import { Badge, type BadgeProps } from '@/components/ui/badge'

import {
  CHECK_LABELS,
  HIDDEN_AI_BLOCKED,
  HIDDEN_AI_REVIEW,
  LEVEL_LABELS,
  NO_RESULT,
  PRIVATE_INSTALLS_ANYWAY,
  STATUS_LABELS,
  VERDICT_LABELS,
  staleCloneWords,
} from './theme-labels'

const KIB = 1024
const UNKNOWN = 'unknown'
const NOT_RUN = 'not_run'
const STATUS_ABSENT = 'absent'

interface BadgeLook {
  readonly variant: NonNullable<BadgeProps['variant']>
  readonly className?: string
}

/** By result: pass solid, warning amber, error red, not measured (and every v1 non-pass) muted. */
const RESULT_BADGES: Readonly<Record<string, BadgeLook>> = {
  pass: { variant: 'default' },
  warning: { variant: 'outline', className: 'border-transparent bg-warning text-warning-foreground' },
  error: { variant: 'destructive' },
  not_measured: { variant: 'secondary' },
}
const MUTED_BADGE: BadgeLook = { variant: 'secondary' }

/**
 * The verdict's own cause among the checks (review_stage.verdict_of): a warning for
 * `review_required`, an error for `blocked`. When no check shown has it, the AI code read
 * (left out by member_view) decided the verdict.
 */
const VERDICT_CAUSES: Readonly<Record<string, { readonly level: string; readonly note: string }>> = {
  review_required: { level: 'warning', note: HIDDEN_AI_REVIEW },
  blocked: { level: 'error', note: HIDDEN_AI_BLOCKED },
}

export interface ScanCheck {
  readonly key: string
  readonly label: string
  /** The check's v2 `level` when it has one, else its v1 `status`, else `unknown`. */
  readonly result: string
  readonly resultLabel: string
  readonly reason: string | null
  /** What the check measured, in one line. */
  readonly detail: string | null
  /** What it found: its `problems`, and each stale clone of `clonedFrom`. */
  readonly problems: readonly string[]
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

function kib(value: unknown): string | null {
  return typeof value === 'number' ? `${(value / KIB).toFixed(1)} KB` : null
}

/** What the sizes check measured, in one line. */
function sizesDetail(check: Record<string, unknown>): string | null {
  const parts = [
    ['package', kib(check.packageBytes)],
    ['code', kib(check.codeBytes)],
    ['entry', kib(check.entryBytes)],
  ].filter((part): part is [string, string] => part[1] !== null)
  return parts.length ? parts.map(([name, size]) => `${name} ${size}`).join(' · ') : null
}

/**
 * services/themes/mobile_contrast.py SUBJECT_DEFAULTS: under `fromLooks` the contrast check measures the
 * prepaint table's defaults first (id `boot/prepaint.json#defaults`); they are no preset, so not listed.
 */
const DEFAULTS_KIND = 'defaults'

function isTableDefaults(preset: unknown): boolean {
  return isRecord(preset) && preset.kind === DEFAULTS_KIND
}

function idsOf(list: unknown): string[] {
  const items = Array.isArray(list) ? list.filter((item) => !isTableDefaults(item)) : []
  return items.map((item) => (isRecord(item) ? item.id : item)).filter((id): id is string => typeof id === 'string')
}

function listed(prefix: string, names: readonly string[]): string | null {
  return names.length ? `${prefix}: ${names.join(', ')}` : null
}

/** The looks and contrast checks: the presets, or the mobile block's schemes (mobile_contrast.py). */
function subjectsDetail(check: Record<string, unknown>): string | null {
  return listed('Presets', idsOf(check.presets)) ?? listed('Schemes', idsOf(check.schemes))
}

/** sdk_uses.uses_check: the SDK areas the code uses. */
/** The SDK areas the code uses, and the ones theme.json declares that it never touches (sdk_uses.py: reported, not warned). */
function usesDetail(check: Record<string, unknown>): string | null {
  const lines = [listed('Areas used', strings(check.usedAreas)), listed('Declared, not used', strings(check.declaredUnused))]
  return lines.filter((line): line is string => line !== null).join(' · ') || null
}

/** review_stage.clone_check: the routes the theme copies a core page for. */
function clonesDetail(check: Record<string, unknown>): string | null {
  return listed('Cloned pages', strings(check.clones))
}

/** scan.py's cssScope is `absent` for a package with no style sheet; under a level its badge reads Passed. */
function cssScopeDetail(check: Record<string, unknown>): string | null {
  return text(check.level) && check.status === STATUS_ABSENT ? STATUS_LABELS[STATUS_ABSENT] : null
}

const DETAILS: Readonly<Record<string, (check: Record<string, unknown>) => string | null>> = {
  sizes: sizesDetail,
  cssScope: cssScopeDetail,
  uses: usesDetail,
  clonedFrom: clonesDetail,
}

/** drift.stale_clones: `{route, path, sha, releaseSha}`, `releaseSha` null when the path is gone. */
function staleClones(value: unknown): string[] {
  const stale = Array.isArray(value) ? value.filter(isRecord) : []
  return stale.map((clone) => staleCloneWords(text(clone.route), text(clone.path), text(clone.releaseSha) !== null))
}

function resultOf(check: Record<string, unknown>): Pick<ScanCheck, 'result' | 'resultLabel'> {
  const level = text(check.level)
  if (level) return { result: level, resultLabel: LEVEL_LABELS[level] ?? level }
  const status = text(check.status)
  if (status) return { result: status, resultLabel: STATUS_LABELS[status] ?? status }
  return { result: UNKNOWN, resultLabel: NO_RESULT }
}

function checkOf(key: string, value: unknown): ScanCheck {
  const check = isRecord(value) ? value : {}
  return {
    key,
    label: CHECK_LABELS[key] ?? key,
    ...resultOf(check),
    reason: text(check.reason),
    detail: (DETAILS[key] ?? subjectsDetail)(check),
    problems: [...strings(check.problems), ...staleClones(check.stale)],
  }
}

/** The report's checks in the server's order; an empty list for a report this page can't read. */
export function scanChecksOf(report: unknown): ScanCheck[] {
  const checks = isRecord(report) && isRecord(report.checks) ? report.checks : {}
  return Object.entries(checks).map(([key, value]) => checkOf(key, value))
}

export function verdictLabel(verdict: string | null): string {
  const value = verdict || NOT_RUN
  return VERDICT_LABELS[value] ?? value
}

/**
 * The line under a Needs a reviewer or Blocked verdict: that the AI code read decided it, when no check shown
 * explains it; else that a private upload installs whatever the verdict. Null for any other verdict.
 */
export function hiddenCheckNote(verdict: string | null, checks: readonly ScanCheck[]): string | null {
  const cause = verdict ? VERDICT_CAUSES[verdict] : undefined
  if (!cause || checks.length === 0) return null
  return checks.some((check) => check.result === cause.level) ? PRIVATE_INSTALLS_ANYWAY : cause.note
}

function CheckRow({ check }: { check: ScanCheck }) {
  const look = RESULT_BADGES[check.result] ?? MUTED_BADGE
  return (
    <li className="rounded-lg border border-border/50 px-3 py-2 text-sm" data-testid={`theme-scan-check-${check.key}`}>
      <div className="flex items-center justify-between gap-2">
        <span>{check.label}</span>
        <Badge variant={look.variant} className={look.className} data-result={check.result}>
          {check.resultLabel}
        </Badge>
      </div>
      {check.detail && <p className="mt-1 text-xs text-muted-foreground">{check.detail}</p>}
      {check.reason && <p className="mt-1 text-xs text-muted-foreground">{check.reason}</p>}
      {check.problems.length > 0 && (
        <ul className="mt-1 list-disc pl-4 text-xs text-muted-foreground" data-testid={`theme-scan-problems-${check.key}`}>
          {check.problems.map((problem, index) => (
            <li key={`${index}-${problem}`}>{problem}</li>
          ))}
        </ul>
      )}
    </li>
  )
}

export function ScanReport({ report, verdict }: { report: unknown; verdict: string | null }) {
  const checks = scanChecksOf(report)
  const note = hiddenCheckNote(verdict, checks)
  return (
    <section aria-label="Scan report" className="space-y-2" data-testid="theme-scan-report">
      <p className="text-sm font-medium text-foreground">
        Scan report
        <span className="ml-2 text-xs font-normal text-muted-foreground" data-testid="theme-scan-verdict">
          Verdict: {verdictLabel(verdict)}
        </span>
      </p>
      {note && (
        <p className="text-xs text-muted-foreground" data-testid="theme-scan-hidden-check">
          {note}
        </p>
      )}
      {checks.length === 0 ? (
        <p className="text-xs text-muted-foreground">This upload has no scan report.</p>
      ) : (
        <ul className="space-y-1.5">
          {checks.map((check) => (
            <CheckRow key={check.key} check={check} />
          ))}
        </ul>
      )}
    </section>
  )
}
