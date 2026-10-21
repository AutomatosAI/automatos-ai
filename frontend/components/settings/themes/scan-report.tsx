/**
 * PRD-261 A-07d — the scan report of a private upload (PRD §5), as `services/themes/scan.py`
 * stores it: `{ schema, verdict, checks: { sizes, cssScope, looks, contrast, automated } }`.
 *
 * Every check the server ran is shown with what it measured; a check that did not run says so with
 * the server's reason, never as passed. The report is data from the server, read defensively: an
 * unknown check is listed by its key, an unknown status by its value.
 */
import { Badge } from '@/components/ui/badge'

export const CHECK_LABELS: Readonly<Record<string, string>> = {
  sizes: 'Sizes',
  cssScope: 'Style sheet scope',
  looks: 'Looks',
  contrast: 'Contrast (AA)',
  automated: 'Automated review',
}

export const STATUS_LABELS: Readonly<Record<string, string>> = {
  pass: 'Passed',
  absent: 'No style sheet',
  not_run: 'Not run',
}

const KIB = 1024

interface ScanCheck {
  readonly key: string
  readonly status: string
  readonly reason: string | null
  readonly detail: string | null
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
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

function presetsDetail(check: Record<string, unknown>): string | null {
  const presets = Array.isArray(check.presets) ? check.presets.filter((preset) => !isTableDefaults(preset)) : []
  const names = presets.map((preset) => (isRecord(preset) ? preset.id : preset)).filter((id) => typeof id === 'string')
  return names.length ? `Presets: ${names.join(', ')}` : null
}

/** The report's checks in the server's order; an empty list for a report this page can't read. */
export function scanChecksOf(report: unknown): ScanCheck[] {
  const checks = isRecord(report) && isRecord(report.checks) ? report.checks : {}
  return Object.entries(checks).map(([key, value]) => {
    const check = isRecord(value) ? value : {}
    return {
      key,
      status: typeof check.status === 'string' ? check.status : 'unknown',
      reason: typeof check.reason === 'string' ? check.reason : null,
      detail: key === 'sizes' ? sizesDetail(check) : presetsDetail(check),
    }
  })
}

export function ScanReport({ report, verdict }: { report: unknown; verdict: string | null }) {
  const checks = scanChecksOf(report)
  return (
    <section aria-label="Scan report" className="space-y-2" data-testid="theme-scan-report">
      <p className="text-sm font-medium text-foreground">
        Scan report
        <span className="ml-2 text-xs font-normal text-muted-foreground">
          Verdict: {STATUS_LABELS[verdict ?? 'not_run'] ?? verdict}
        </span>
      </p>
      {checks.length === 0 ? (
        <p className="text-xs text-muted-foreground">This upload has no scan report.</p>
      ) : (
        <ul className="space-y-1.5">
          {checks.map((check) => (
            <li key={check.key} className="rounded-lg border border-border/50 px-3 py-2 text-sm">
              <div className="flex items-center justify-between gap-2">
                <span>{CHECK_LABELS[check.key] ?? check.key}</span>
                <Badge variant={check.status === 'pass' ? 'default' : 'secondary'}>
                  {STATUS_LABELS[check.status] ?? check.status}
                </Badge>
              </div>
              {check.detail && <p className="mt-1 text-xs text-muted-foreground">{check.detail}</p>}
              {check.reason && <p className="mt-1 text-xs text-muted-foreground">{check.reason}</p>}
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}
