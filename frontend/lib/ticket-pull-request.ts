/**
 * Ticket PR fields — the pull request a coding session reported on its ticket.
 *
 * A Claude Code session that opens a pull request says so through its `update_ticket` tool
 * (orchestrator services/ticket_pull_request.py); the board's API serves it as `pull_request`,
 * already re-checked. It is checked once more here before anything is shown: the owner approves
 * the card, and with the install's setting on that Approve merges whatever pull request the card
 * shows, so only an https address is ever a link, and its text names the repository and number
 * the merge would act on. The Approve sends back the pull request it showed (its address and the
 * `at` stamp of its last write: `approvalIdentity`), and the server merges nothing that changed
 * since; the address is kept exactly as the server sent it, so that comparison is exact.
 */

export interface PullRequestCommit {
  sha: string
  subject: string
}

export interface TicketPullRequest {
  /** The pull request's own https address, or null when the session reported none. */
  url: string | null
  /** owner/repo (a GitLab project's path), from the address. */
  repository: string | null
  number: number | null
  branch: string | null
  commits: PullRequestCommit[]
  /** What the session ran or checked and what it saw, or that nothing could be run. */
  tested: string | null
  /** When it was last written: sent back with an Approve, so a pull request changed since is not merged. */
  at: string | null
  /** The server's word: a person's Approve would merge it (the install's setting on, its repository listed). */
  merges_on_approve: boolean
}

/** What an Approve sends back about the pull request it showed. */
export interface ApprovalPullRequest {
  url: string
  at: string | null
}

export const PULL_REQUEST_SECTION = 'Pull request'
export const PR_CHIP = 'PR'
export const PR_CHIP_TITLE = 'Its session opened a pull request: open the ticket to see it'
export const BRANCH_LABEL = 'Branch'
export const COMMITS_LABEL = 'Commits'
export const TESTED_LABEL = 'Tested'
export const OPEN_PULL_REQUEST = 'Opens the pull request in a new tab'
export const APPROVE_WILL_MERGE = 'Approving will merge this pull request (squash) once its checks pass.'
export const REMOVE_PULL_REQUEST = 'Remove'
export const REMOVE_PULL_REQUEST_TITLE = 'Take this pull request off the ticket?'
export const REMOVE_PULL_REQUEST_BODY =
  'The ticket forgets it: nothing changes on GitHub, and approving the ticket merges nothing.'
export const REMOVE_PULL_REQUEST_CONFIRM = 'Remove it'
export const PULL_REQUEST_REMOVED = 'The pull request is off the ticket.'
export const PULL_REQUEST_NOT_REMOVED = 'Could not take the pull request off the ticket'
/** How much of a sha the card shows (git's own short form). */
export const SHORT_SHA_CHARS = 7

const SHA = /^[0-9a-f]{7,40}$/i

function text(value: unknown): string | null {
  return typeof value === 'string' && value.trim() ? value.trim() : null
}

function httpsUrl(value: unknown): string | null {
  const raw = text(value)
  if (!raw) return null
  try {
    const parsed = new URL(raw)
    return parsed.protocol === 'https:' && !parsed.username && !parsed.password ? raw : null
  } catch {
    return null
  }
}

function commitsOf(value: unknown): PullRequestCommit[] {
  if (!Array.isArray(value)) return []
  return value
    .filter((c): c is Record<string, unknown> => !!c && typeof c === 'object' && typeof c.sha === 'string' && SHA.test(c.sha))
    .map((c) => ({ sha: String(c.sha).toLowerCase(), subject: typeof c.subject === 'string' ? c.subject : '' }))
}

/** The API's `pull_request`, checked; null when the ticket has none worth showing. */
export function pullRequestOf(raw: unknown): TicketPullRequest | null {
  if (!raw || typeof raw !== 'object') return null
  const item = raw as Record<string, unknown>
  const url = httpsUrl(item.url)
  const number = typeof item.number === 'number' && Number.isInteger(item.number) && item.number > 0 ? item.number : null
  const pr: TicketPullRequest = {
    url,
    repository: url ? text(item.repository) : null,
    number: url ? number : null,
    branch: text(item.branch),
    commits: commitsOf(item.commits),
    tested: text(item.tested),
    at: text(item.at),
    merges_on_approve: !!url && item.merges_on_approve === true,
  }
  return pr.url || pr.branch || pr.commits.length > 0 || pr.tested ? pr : null
}

/** "owner/repo #12", or the address itself when it was not taken apart. */
export function pullRequestLinkText(pr: Pick<TicketPullRequest, 'url' | 'repository' | 'number'>): string {
  if (pr.repository && pr.number) return `${pr.repository} #${pr.number}`
  return pr.url ?? ''
}

export function shortSha(sha: string): string {
  return sha.slice(0, SHORT_SHA_CHARS)
}

/** The pull request an Approve showed, as the approve request carries it; null when it showed none. */
export function approvalIdentity(pr: TicketPullRequest | null | undefined): ApprovalPullRequest | null {
  return pr?.url ? { url: pr.url, at: pr.at } : null
}
