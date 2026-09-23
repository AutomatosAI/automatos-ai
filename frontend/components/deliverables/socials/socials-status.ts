/**
 * PRD-251 Wave 0 — the Socials tab's reading of the S0.3 rules: status labels,
 * the order the groups show in, and which actions a role and a status allow.
 * The server enforces all of it; this only decides which buttons to show.
 */
import type { SocialPost, SocialPostStatus } from '@/lib/api-client'
import type { Workspace } from '@/components/workspace-provider'

type WorkspaceRole = Workspace['role']

// The service's limits (modules/socials/service.py TITLE_MAX_CHARS / COMMENT_MAX_CHARS).
export const SOCIAL_POST_TITLE_MAX_CHARS = 500
export const SOCIAL_COMMENT_MAX_CHARS = 2000

/** review_log actions (modules/socials/service.py ACTION_*) as the history shows them. */
export const REVIEW_ACTION_LABELS: Record<string, string> = {
  submit: 'Sent for approval',
  approve: 'Approved',
  request_changes: 'Changes requested',
  reject: 'Rejected',
  schedule: 'Scheduled',
  unschedule: 'Unscheduled',
  edit: 'Edited',
  approval_voided: 'Approval voided by an edit',
}

export const SOCIAL_STATUS_LABELS: Record<SocialPostStatus, string> = {
  draft: 'Draft',
  rendering: 'Rendering',
  needs_approval: 'Needs approval',
  changes_requested: 'Changes requested',
  approved: 'Approved',
  scheduled: 'Scheduled',
  publishing: 'Publishing',
  published: 'Published',
  partially_published: 'Partially published',
  failed: 'Failed',
  missed: 'Missed',
  archived: 'Archived',
}

/** Group order: what needs a person first, finished and archived work last. */
export const SOCIAL_STATUS_ORDER: ReadonlyArray<SocialPostStatus> = [
  'needs_approval',
  'changes_requested',
  'draft',
  'rendering',
  'approved',
  'scheduled',
  'publishing',
  'failed',
  'missed',
  'partially_published',
  'published',
  'archived',
]

// The server's role matrix (modules/policy/roles.py): owner and admin hold
// workspace:manage; owner, admin and editor hold documents:create/update and
// socials:approve; a viewer only reads.
const MANAGE_ROLES: ReadonlySet<WorkspaceRole> = new Set<WorkspaceRole>(['owner', 'admin'])
const AUTHOR_ROLES: ReadonlySet<WorkspaceRole> = new Set<WorkspaceRole>(['owner', 'admin', 'editor'])
const REVIEW_ROLES: ReadonlySet<WorkspaceRole> = new Set<WorkspaceRole>(['owner', 'admin', 'editor'])

// The S0.3 status machine, seen from the buttons.
const SUBMITTABLE: ReadonlySet<SocialPostStatus> = new Set<SocialPostStatus>(['draft', 'changes_requested'])
const REVIEWABLE: ReadonlySet<SocialPostStatus> = new Set<SocialPostStatus>(['needs_approval'])
const EDITABLE: ReadonlySet<SocialPostStatus> = new Set<SocialPostStatus>([
  'draft', 'needs_approval', 'changes_requested', 'approved', 'scheduled',
])

export function canTurnOnSocials(role: WorkspaceRole | undefined): boolean {
  return !!role && MANAGE_ROLES.has(role)
}

export function canAuthorPosts(role: WorkspaceRole | undefined): boolean {
  return !!role && AUTHOR_ROLES.has(role)
}

export interface PostActions {
  edit: boolean
  submit: boolean
  review: boolean
}

/** The actions `role` may take on a post in `status`. */
export function postActions(role: WorkspaceRole | undefined, status: SocialPostStatus): PostActions {
  const author = canAuthorPosts(role)
  return {
    edit: author && EDITABLE.has(status),
    submit: author && SUBMITTABLE.has(status),
    review: !!role && REVIEW_ROLES.has(role) && REVIEWABLE.has(status),
  }
}

export interface StatusGroup {
  status: SocialPostStatus
  label: string
  posts: SocialPost[]
}

function newestFirst(a: SocialPost, b: SocialPost): number {
  return (b.created_at || '').localeCompare(a.created_at || '')
}

/** Posts grouped by status in SOCIAL_STATUS_ORDER, newest first; empty groups omitted. */
export function groupPostsByStatus(posts: ReadonlyArray<SocialPost>): StatusGroup[] {
  return SOCIAL_STATUS_ORDER.map((status) => ({
    status,
    label: SOCIAL_STATUS_LABELS[status],
    posts: posts.filter((post) => post.status === status).sort(newestFirst),
  })).filter((group) => group.posts.length > 0)
}
