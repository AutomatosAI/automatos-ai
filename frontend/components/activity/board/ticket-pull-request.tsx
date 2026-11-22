'use client'

/**
 * Ticket PR fields — the pull request a coding session opened, on its card and in its viewer.
 * The viewer's section links it (a new tab, no opener), names its branch, lists its commits, says
 * what the session tested and, when the server says so, that approving will merge it; the owner can
 * take a wrong one off (Remove, behind the house confirm). The card in the board list carries a
 * small PR chip.
 */
import { useState } from 'react'
import { GitPullRequest } from 'lucide-react'
import { toast } from 'sonner'
import { DeleteConfirmation } from '@/components/shared/delete-confirmation'
import { useRemovePullRequest } from '@/hooks/use-board-tasks'
import {
  APPROVE_WILL_MERGE,
  BRANCH_LABEL,
  COMMITS_LABEL,
  OPEN_PULL_REQUEST,
  PR_CHIP,
  PR_CHIP_TITLE,
  PULL_REQUEST_NOT_REMOVED,
  PULL_REQUEST_REMOVED,
  PULL_REQUEST_SECTION,
  REMOVE_PULL_REQUEST,
  REMOVE_PULL_REQUEST_BODY,
  REMOVE_PULL_REQUEST_CONFIRM,
  REMOVE_PULL_REQUEST_TITLE,
  TESTED_LABEL,
  pullRequestLinkText,
  shortSha,
  type TicketPullRequest,
} from '@/lib/ticket-pull-request'

export function TicketPullRequestSection({ pullRequest, taskId }: { pullRequest?: TicketPullRequest | null; taskId?: string }) {
  if (!pullRequest) return null
  const { url, branch, commits, tested } = pullRequest
  return (
    <div className="mb-6" data-testid="ticket-pull-request">
      <div className="flex items-center gap-1.5 mb-2">
        <GitPullRequest className="w-3 h-3 text-muted-foreground" />
        <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">{PULL_REQUEST_SECTION}</h4>
        {taskId && <RemovePullRequest taskId={taskId} />}
      </div>
      <div className="glass-card rounded-lg p-4 text-sm space-y-3">
        {url && (
          <a
            href={url}
            target="_blank"
            rel="noopener noreferrer"
            title={OPEN_PULL_REQUEST}
            className="inline-flex items-center gap-1.5 font-medium text-primary hover:underline break-all"
          >
            <GitPullRequest className="w-3.5 h-3.5 shrink-0" />
            {pullRequestLinkText(pullRequest)}
          </a>
        )}
        {pullRequest.merges_on_approve && (
          <p className="text-xs text-muted-foreground" data-testid="pull-request-merges-on-approve">{APPROVE_WILL_MERGE}</p>
        )}
        {branch && (
          <p className="text-xs">
            <span className="text-muted-foreground">{BRANCH_LABEL} </span>
            <code className="font-mono break-all">{branch}</code>
          </p>
        )}
        {commits.length > 0 && (
          <div>
            <p className="text-xs text-muted-foreground mb-1">{COMMITS_LABEL}</p>
            <ul className="text-xs space-y-0.5 max-h-40 overflow-y-auto" data-testid="pull-request-commits">
              {commits.map((c) => (
                <li key={c.sha} className="truncate" title={`${c.sha} ${c.subject}`}>
                  <code className="font-mono text-muted-foreground">{shortSha(c.sha)}</code> {c.subject}
                </li>
              ))}
            </ul>
          </div>
        )}
        {tested && (
          <div>
            <p className="text-xs text-muted-foreground mb-1">{TESTED_LABEL}</p>
            <p className="text-xs whitespace-pre-wrap" data-testid="pull-request-tested">{tested}</p>
          </div>
        )}
      </div>
    </div>
  )
}

/** The owner's Remove: takes a wrong pull request off the ticket, after the house confirm (PRD-169: no window.confirm). */
function RemovePullRequest({ taskId }: { taskId: string }) {
  const [asking, setAsking] = useState(false)
  const remove = useRemovePullRequest()
  const removeIt = async () => {
    try {
      await remove.mutateAsync({ taskId })
      toast.success(PULL_REQUEST_REMOVED)
    } catch (err) {  // the server's words (a person only; none to remove), or ours
      toast.error(err instanceof Error && err.message ? err.message : PULL_REQUEST_NOT_REMOVED)
    }
  }
  return (
    <>
      <button
        type="button"
        className="ml-auto text-[11px] text-muted-foreground underline-offset-2 hover:text-destructive hover:underline"
        onClick={() => setAsking(true)}
      >
        {REMOVE_PULL_REQUEST}
      </button>
      <DeleteConfirmation
        open={asking}
        onOpenChange={setAsking}
        title={REMOVE_PULL_REQUEST_TITLE}
        description={REMOVE_PULL_REQUEST_BODY}
        confirmLabel={REMOVE_PULL_REQUEST_CONFIRM}
        onConfirm={removeIt}
      />
    </>
  )
}

/** The board list's mark for a ticket with a pull request. */
export function PullRequestChip({ pullRequest }: { pullRequest?: TicketPullRequest | null }) {
  if (!pullRequest?.url) return null
  return (
    <span
      className="flex items-center gap-0.5 text-[10px] font-medium px-1.5 py-0.5 rounded bg-[hsl(var(--success))]/10 text-[hsl(var(--success))]"
      title={PR_CHIP_TITLE}
      data-testid="pull-request-chip"
    >
      <GitPullRequest className="w-2.5 h-2.5" />
      {PR_CHIP}
    </span>
  )
}
