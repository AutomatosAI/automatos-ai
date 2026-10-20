/**
 * PRD-261 C-03b — a theme version's scan report as the reviewer reads it: each check with its
 * level (Error, Warning, Pass, Not measured), the server's reason and what it found. The LLM check
 * that did not read all of the code says "LLM stage did not run" (`review_stage.py`), never Pass.
 *
 * Also the author's view of a Marketplace submission (C-01b's Publish theme result): the same v2
 * report through `review_stage.member_view`, which leaves the LLM check out.
 *
 * Everything here is server data shown as text (React escapes it); a finding's evidence is code
 * from the submitted package and is shown in a <code> element, never run or linked.
 */
import { Badge } from '@/components/ui/badge'
import { cn } from '@/lib/utils'

import {
  CHECK_LABELS,
  LEVEL_LABELS,
  LLM_DID_NOT_RUN,
  VERDICT_LABELS,
  labelOf,
  reportChecksOf,
  type ReportCheck,
  type ReportFinding,
} from './review-words'

const LEVEL_CLASSES: Readonly<Record<string, string>> = {
  error: 'border-destructive/40 bg-destructive/10 text-destructive',
  warning: 'border-warning/40 bg-warning/10 text-warning',
  pass: 'border-success/40 bg-success/10 text-success',
}

export function LevelBadge({ level }: { level: string }) {
  return (
    <Badge variant="outline" className={cn(LEVEL_CLASSES[level])} data-level={level}>
      {labelOf(LEVEL_LABELS, level)}
    </Badge>
  )
}

function FindingRow({ finding }: { finding: ReportFinding }) {
  return (
    <li className="rounded border border-border/50 px-2 py-1 text-xs">
      <span className="font-medium">{finding.severity}</span> · {finding.category}
      {finding.file && <span className="text-muted-foreground"> · {finding.file}</span>}
      {finding.description && <p className="mt-0.5">{finding.description}</p>}
      {finding.evidence && <code className="mt-0.5 block whitespace-pre-wrap break-all">{finding.evidence}</code>}
    </li>
  )
}

function LlmDetail({ check }: { check: ReportCheck }) {
  return (
    <div className="mt-1 space-y-1 text-xs">
      {check.llmDidNotRun && (
        <p className="font-medium text-warning" data-testid="theme-llm-not-run">
          {check.reason ?? LLM_DID_NOT_RUN}
        </p>
      )}
      {check.riskScore !== null && <p className="text-muted-foreground">Risk score: {check.riskScore}</p>}
      {check.summary && <p className="text-muted-foreground">{check.summary}</p>}
      {check.findings.length > 0 && (
        <ul className="space-y-1">
          {check.findings.map((finding, index) => (
            <FindingRow key={`${finding.file ?? ''}-${index}`} finding={finding} />
          ))}
        </ul>
      )}
    </div>
  )
}

function CheckRow({ check }: { check: ReportCheck }) {
  const isLlm = check.key === 'llm'
  return (
    <li className="rounded-lg border border-border/50 px-3 py-2 text-sm" data-testid={`theme-check-${check.key}`}>
      <div className="flex items-center justify-between gap-2">
        <span>{CHECK_LABELS[check.key] ?? check.key}</span>
        <LevelBadge level={check.level} />
      </div>
      {!(isLlm && check.llmDidNotRun) && check.reason && (
        <p className="mt-1 text-xs text-muted-foreground">{check.reason}</p>
      )}
      {check.problems.length > 0 && (
        <ul className="mt-1 list-disc pl-4 text-xs text-muted-foreground">
          {check.problems.map((problem, index) => (
            <li key={`${index}-${problem}`}>{problem}</li>
          ))}
        </ul>
      )}
      {isLlm && <LlmDetail check={check} />}
    </li>
  )
}

interface ScanChecksProps {
  readonly report: unknown
  readonly verdict: string | null
  /** The verdict's words; the Publish theme result (C-01b) passes the author's. */
  readonly verdictLabels?: Readonly<Record<string, string>>
}

export function ScanChecks({ report, verdict, verdictLabels = VERDICT_LABELS }: ScanChecksProps) {
  const checks = reportChecksOf(report)
  return (
    <section aria-label="Scan report" className="space-y-2" data-testid="theme-review-scan">
      <p className="text-sm font-medium text-foreground">
        Scan report
        <span className="ml-2 text-xs font-normal text-muted-foreground" data-testid="theme-review-verdict">
          Verdict: {labelOf(verdictLabels, verdict)}
        </span>
      </p>
      {checks.length === 0 ? (
        <p className="text-xs text-muted-foreground">This version has no scan report.</p>
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
