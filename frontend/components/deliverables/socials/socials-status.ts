/**
 * PRD-251 Wave 0 — the Socials tab's reading of the S0.3 rules: status labels,
 * the order the groups show in, and which actions a role and a status allow.
 * Wave 1 (S1.1c) adds rendering: a post with a template renders from any status
 * that holds no approval, and a failed render can be edited and rendered again.
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
  // US-116: an agent drafted the post; the entry's comment names the agent.
  draft: 'Drafted',
  submit: 'Sent for approval',
  approve: 'Approved',
  request_changes: 'Changes requested',
  reject: 'Rejected',
  schedule: 'Scheduled',
  unschedule: 'Unscheduled',
  edit: 'Edited',
  approval_voided: 'Approval voided by an edit',
  render: 'Render started',
  render_done: 'Rendered',
  render_failed: 'Render failed',
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
  'draft', 'needs_approval', 'changes_requested', 'approved', 'scheduled', 'failed',
])
// S1.1c: the statuses a render starts from (modules/socials/service.py TRANSITIONS['render']).
const RENDERABLE: ReadonlySet<SocialPostStatus> = new Set<SocialPostStatus>([
  'draft', 'changes_requested', 'needs_approval', 'failed',
])

export function canTurnOnSocials(role: WorkspaceRole | undefined): boolean {
  return !!role && MANAGE_ROLES.has(role)
}

export function canAuthorPosts(role: WorkspaceRole | undefined): boolean {
  return !!role && AUTHOR_ROLES.has(role)
}

/** PRD-251 S1.3: the brand kit is edited by whoever holds workspace:manage. */
export function canEditBrandKit(role: WorkspaceRole | undefined): boolean {
  return !!role && MANAGE_ROLES.has(role)
}

export interface PostActions {
  edit: boolean
  submit: boolean
  review: boolean
  render: boolean
}

/** The actions `role` may take on a post in `status`; only a post with a template renders. */
export function postActions(
  role: WorkspaceRole | undefined,
  status: SocialPostStatus,
  hasTemplate = false,
): PostActions {
  const author = canAuthorPosts(role)
  return {
    edit: author && EDITABLE.has(status),
    submit: author && SUBMITTABLE.has(status),
    review: !!role && REVIEW_ROLES.has(role) && REVIEWABLE.has(status),
    render: author && hasTemplate && RENDERABLE.has(status),
  }
}

// S1.5: the post formats that render as a still, with no voice to choose.
const STILL_FORMATS: ReadonlySet<string> = new Set(['image', 'carousel', 'fact_card', 'infographic'])

/** Whether a post renders with a spoken script, so it has a voice to choose (S1.5): a
 * post with a template whose format is video, or not chosen yet. */
export function speaksAScript(post: Pick<SocialPost, 'template_id' | 'format'>): boolean {
  return !!post.template_id && !(post.format && STILL_FORMATS.has(post.format))
}

/** Whether any post is rendering: the list polls until none is. */
export function anyRendering(posts: ReadonlyArray<SocialPost>): boolean {
  return posts.some((post) => post.status === 'rendering')
}

/** "3.5 / 10" (or "3.5" with no quota), minutes to one decimal place. */
export function formatRenderMinutes(used: number, quota: number | null): string {
  const fmt = (value: number) => (Number.isInteger(value) ? String(value) : value.toFixed(1))
  return quota === null ? fmt(used) : `${fmt(used)} / ${fmt(quota)}`
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
