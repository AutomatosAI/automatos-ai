/**
 * PRD-251 S2.2 (US-207..US-209) — the composer's data: the proposal as the
 * person edits it, the post it saves as, and the targets it writes.
 * The server checks all of it again (the post's validators, the channel registry).
 */
import type {
  CreateSocialPostInput,
  SocialChannel,
  SocialComposeProposal,
  SocialComposeTemplate,
  SocialCopyLimits,
  SocialPostKind,
  SocialPostTargetInput,
  SocialPostTargetOptions,
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
  /** The template's variables and sizes (US-208): what the fields come from. */
  template: SocialComposeTemplate | null
  variables: SocialComposeProposal['variables']
  sources: SocialComposeProposal['sources']
  /** Channel (toolkit) → the post kind it publishes. */
  kinds: Record<string, SocialPostKind>
  /** Channel (toolkit) → what its post kind needs (US-209: YouTube's privacy and category). */
  options: Record<string, SocialPostTargetOptions>
  warnings: string[]
}

// The post kinds a post format publishes as, most fitting first.
const FORMAT_KINDS: Record<string, SocialPostKind[]> = {
  video: ['video', 'reel', 'short'],
  image: ['image'],
  carousel: ['carousel', 'image'],
  fact_card: ['image'],
  infographic: ['image'],
  // PRD-251B US-B109: a text-only post (X and LinkedIn take one).
  text: ['text'],
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
    template: proposal.template ?? null,
    variables: { ...proposal.variables },
    sources: { ...proposal.sources },
    kinds,
    options: {},
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
export function draftTargets(draft: Pick<ComposerDraft, 'kinds' | 'options'>): SocialPostTargetInput[] {
  return Object.entries(draft.kinds).map(([toolkit, post_kind]) => {
    const options = draft.options[toolkit]
    return options && Object.keys(options).length > 0 ? { toolkit, post_kind, options } : { toolkit, post_kind }
  })
}

/** Whether the draft renders as a video: a preview render (US-208), else the real one. */
export function isVideoDraft(draft: Pick<ComposerDraft, 'format' | 'template'>): boolean {
  return draft.template ? draft.template.format === 'social_video' : draft.format === 'video'
}

const HASHTAG = /(?<![\w#])#\w+/g

export interface CopyCount {
  count: number
  limit: number | null
  hashtags: number
  hashtagLimit: number | null
  over: boolean
}

/** A channel's copy against its limits (US-207's): characters, and Instagram's hashtags. */
export function copyCount(text: string, limits: SocialCopyLimits | null | undefined): CopyCount {
  const hashtags = (text.match(HASHTAG) ?? []).length
  const limit = limits?.text ?? null
  const hashtagLimit = limits?.hashtags ?? null
  const over = (limit !== null && text.length > limit) || (hashtagLimit !== null && hashtags > hashtagLimit)
  return { count: text.length, limit, hashtags, hashtagLimit, over }
}

/** The chosen channels whose copy is over their limits: submit waits until there are none. */
export function channelsOverLimit(
  draft: Pick<ComposerDraft, 'kinds' | 'perChannel' | 'base'>,
  channels: ReadonlyArray<SocialChannel>,
): string[] {
  return Object.keys(draft.kinds).filter((toolkit) => {
    const channel = channels.find((c) => c.toolkit === toolkit)
    return copyCount(draft.perChannel[toolkit] ?? draft.base, channel?.copy_limits).over
  })
}

/** The draft with `toolkit` chosen (its default kind, the base copy to start) or left out. */
export function withChannel(draft: ComposerDraft, channel: SocialChannel, on: boolean): ComposerDraft {
  const { [channel.toolkit]: _kind, ...kinds } = draft.kinds
  if (!on) return { ...draft, kinds }
  const kind = defaultKind(channel, draft.format)
  if (!kind) return draft
  const perChannel = channel.toolkit in draft.perChannel ? draft.perChannel : { ...draft.perChannel, [channel.toolkit]: draft.base }
  return { ...draft, kinds: { ...kinds, [channel.toolkit]: kind }, perChannel }
}

/** "1080x1920" → "1080×1920 (9:16)": the template's sizes as the aspect ratios the post renders in. */
export function aspectLabel(size: string): string {
  const [w, h] = size.split('x').map(Number)
  if (!w || !h) return size
  const gcd = (a: number, b: number): number => (b === 0 ? a : gcd(b, a % b))
  const d = gcd(w, h)
  return `${w}×${h} (${w / d}:${h / d})`
}
