/**
 * PRD-261 C-01b — what the author sees once the Marketplace accepted a submission.
 *
 * The answer of `POST /api/marketplace/themes/submit` (`submission_item`): the package, stored
 * `pending` and not reviewed, with the automated stage's verdict and its plain checks
 * (`review_stage.member_view`; the LLM check's findings stay with the reviewer). Nothing is listed
 * yet: a person reviews every version (C-03a, D3), and a `blocked` verdict can't be approved. The
 * checks are shown by A-07d's scan report, reused, with the submission's verdict words.
 */
import { Badge } from '@/components/ui/badge'
import { ScanReport } from '@/components/settings/themes/scan-report'
import type { ThemeSubmission } from '@/lib/api-client'

export const PENDING_REVIEW = 'Pending review'
export const REVIEW_NOTE = 'A person at Automatos reviews every version before it is listed in the Marketplace.'
export const DUPLICATE_NOTE = 'These exact files were submitted before: this is that submission.'
export const BLOCKED_NOTE =
  'The automated checks blocked this version, so it can’t be approved. Fix what they found and submit a new version.'

/** services/themes/review_stage.py: pass, review_required, blocked; not_run for a stage that did not run. */
export const SUBMISSION_VERDICTS: Readonly<Record<string, string>> = {
  pass: 'Passed the automated checks',
  review_required: 'Needs a closer look by the reviewer',
  blocked: 'Blocked by the automated checks',
  not_run: 'Automated checks not run',
}

const VERDICT_BLOCKED = 'blocked'
const VERDICT_NOT_RUN = 'not_run'
const STATUS_PENDING = 'pending'

/** The verdict in words; an unknown verdict as the server wrote it. */
export function submissionVerdictLabel(verdict: string | null): string {
  const key = verdict ?? VERDICT_NOT_RUN
  return SUBMISSION_VERDICTS[key] ?? key
}

export function PublishResult({ submission }: { submission: ThemeSubmission }) {
  const pkg = submission.package
  return (
    <div className="space-y-3 rounded-xl border border-border/50 px-4 py-3" data-testid="publish-result">
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-medium text-foreground">{pkg.name || pkg.themeId}</span>
        <span className="text-xs text-muted-foreground">{pkg.themeId}</span>
        <span className="text-xs text-muted-foreground">{pkg.version}</span>
        <Badge variant="secondary">{pkg.status === STATUS_PENDING ? PENDING_REVIEW : pkg.status}</Badge>
      </div>
      <p className="text-xs text-muted-foreground">
        {REVIEW_NOTE}
        {submission.duplicate ? ` ${DUPLICATE_NOTE}` : ''}
      </p>
      {pkg.scanVerdict === VERDICT_BLOCKED && (
        <p className="text-sm text-destructive" data-testid="publish-blocked">
          {BLOCKED_NOTE}
        </p>
      )}
      <ScanReport report={pkg.scanReport} verdict={submissionVerdictLabel(pkg.scanVerdict)} />
    </div>
  )
}
