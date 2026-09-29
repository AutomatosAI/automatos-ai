/**
 * PRD-251 S2.2 (US-207..US-209) — the composer's data: the proposal as the
 * person edits it, the post it saves as, and the targets it writes.
 * The server checks all of it again (the post's validators, the channel registry).
 */
import type {
  CreateSocialPostInput,
  SocialChannel,
  SocialComposeProposal,
  SocialPostKind,
  SocialPostTargetInput,
} from '@/lib/api-client'

/** The proposal as the composer holds it while the person edits. */
export interface ComposerDraft {
  brief: string
  title: string
  base: string
  /** Channel (toolkit) → its own text. */
  perChannel: Record<string, string>
  format: string | null
  templateId: string | null
  variables: SocialComposeProposal['variables']
  sources: SocialComposeProposal['sources']
  /** Channel (toolkit) → the post kind it publishes. */
  kinds: Record<string, SocialPostKind>
  warnings: string[]
}

// The post kinds a post format publishes as, most fitting first.
const FORMAT_KINDS: Record<string, SocialPostKind[]> = {
  video: ['video', 'reel', 'short'],
  image: ['image'],
  carousel: ['carousel', 'image'],
  fact_card: ['image'],
  infographic: ['image'],
}

/** The kind a channel posts a post of `format` as: the first available that fits, else its first available. */
export function defaultKind(channel: SocialChannel, format: string | null): SocialPostKind | null {
  const available = channel.post_kinds.filter((kind) => kind.available).map((kind) => kind.kind)
  const wanted = (format && FORMAT_KINDS[format]) || []
  return wanted.find((kind) => available.includes(kind)) ?? available[0] ?? null
}

/** The composer's draft, from the server's proposal. */
export function draftFromProposal(
  brief: string,
  proposal: SocialComposeProposal,
  channels: ReadonlyArray<SocialChannel>,
): ComposerDraft {
  const kinds: Record<string, SocialPostKind> = {}
  for (const toolkit of proposal.channels) {
    const channel = channels.find((c) => c.toolkit === toolkit)
    const kind = channel ? defaultKind(channel, proposal.format) : null
    if (kind) kinds[toolkit] = kind
  }
  return {
    brief,
    title: proposal.title,
    base: proposal.copy.base,
    perChannel: { ...proposal.copy.per_channel },
    format: proposal.format,
    templateId: proposal.template_id,
    variables: { ...proposal.variables },
    sources: { ...proposal.sources },
    kinds,
    warnings: [...proposal.warnings],
  }
}

/** The post the draft saves as (POST /api/socials/posts). */
export function postInput(draft: ComposerDraft): CreateSocialPostInput {
  return {
    title: draft.title.trim(),
    brief: draft.brief.trim() || null,
    copy: { base: draft.base, channels: { ...draft.perChannel } },
    format: draft.format,
    template_id: draft.templateId,
    variables: draft.variables,
    sources: draft.sources,
  }
}

/** Where the draft publishes (PUT /api/socials/posts/{id}/targets): one target per chosen channel. */
export function draftTargets(draft: ComposerDraft): SocialPostTargetInput[] {
  return Object.entries(draft.kinds).map(([toolkit, post_kind]) => ({ toolkit, post_kind }))
}
