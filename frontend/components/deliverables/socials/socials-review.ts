/**
 * PRD-251 S2.3 (US-206) — what the approval view reads off a post: each
 * channel's copy (the base, or the target's own override), every claim with the
 * source it cites (D7), and whether the newest history entry voided an approval.
 * The server decides all of it again at approve time; this only shows it.
 */
import type { SocialClaimSource, SocialPost } from '@/lib/api-client'

export interface ChannelCopy {
  toolkit: string
  kinds: string[]
  text: string
  /** The channel has its own text, not the base. */
  overridden: boolean
}

/** One entry per target channel, its post kinds and the copy it publishes. */
export function channelCopies(post: Pick<SocialPost, 'copy' | 'targets'>): ChannelCopy[] {
  const base = post.copy?.base ?? ''
  const own = post.copy?.channels ?? {}
  const kinds = new Map<string, string[]>()
  for (const target of post.targets ?? []) {
    kinds.set(target.toolkit, [...(kinds.get(target.toolkit) ?? []), target.post_kind])
  }
  return [...kinds.entries()].map(([toolkit, postKinds]) => ({
    toolkit,
    kinds: postKinds,
    text: typeof own[toolkit] === 'string' ? own[toolkit] : base,
    overridden: typeof own[toolkit] === 'string',
  }))
}

export interface ClaimRow {
  name: string
  value: string
  source: SocialClaimSource | null
}

function isSource(value: unknown): value is SocialClaimSource {
  const source = value as SocialClaimSource | null
  return !!source && typeof source === 'object' && typeof source.kind === 'string' && typeof source.ref === 'string'
}

/** The variables marked `claim: true`, each with the source it cites, or null (D7). */
export function claimRows(post: Pick<SocialPost, 'variables' | 'sources'>): ClaimRow[] {
  const sources = post.sources ?? {}
  return Object.entries(post.variables ?? {})
    .filter(([, spec]) => (spec as { claim?: unknown } | null)?.claim === true)
    .map(([name, spec]) => {
      const value = (spec as { value?: unknown }).value
      const source = sources[name]
      return { name, value: value == null ? '' : String(value), source: isSource(source) ? source : null }
    })
    .sort((a, b) => a.name.localeCompare(b.name))
}

/** The claims with no source: approving them needs the second confirmation. */
export function unsourcedClaims(post: Pick<SocialPost, 'variables' | 'sources'>): string[] {
  return claimRows(post).filter((row) => !row.source).map((row) => row.name)
}

/** Whether the newest history entry is an approval an edit voided (D6). */
/** F256: whether the post goes anywhere. With no channel nothing would post, so it is not approved. */
export function hasChannels(post: Pick<SocialPost, 'targets'>): boolean {
  return (post.targets ?? []).length > 0
}

/** F378: whether the post keeps a take from before Auto's last retake that is not restored yet
 * (the server's `modules/socials/retakes.restoring`). */
export function hasEarlierTake(post: Pick<SocialPost, 'review_log'>): boolean {
  const log = post.review_log ?? []
  const undone = new Set(log.filter((entry) => entry.action === 'retake_undone').map((entry) => entry.undid))
  return log.some((entry) => entry.action === 'retake' && !!entry.previous && !undone.has(entry.at))
}

export function approvalWasVoided(post: Pick<SocialPost, 'review_log'>): boolean {
  const log = post.review_log ?? []
  return log.length > 0 && log[log.length - 1].action === 'approval_voided'
}

/** The claims a 422 UnsourcedClaims answer names (its detail rides the Error's message as JSON). */
export function unsourcedClaimsOf(error: unknown): string[] | null {
  if (!(error instanceof Error)) return null
  try {
    const detail = JSON.parse(error.message) as { claims?: unknown }
    return Array.isArray(detail.claims) ? detail.claims.map(String) : null
  } catch {
    return null
  }
}
