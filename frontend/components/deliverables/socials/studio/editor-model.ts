/**
 * PRD-251B US-B109 — the post editor's data (pure): the post as the person edits it, the
 * fields, channels and slot it saves as, and what each card derives from it (the format's
 * channels and why one cannot take it, the size each channel gets, the ratios the post
 * renders in, the spoken words a length fits). The server checks all of it again.
 * PRD-251C (US-C301): a ticked channel that posts stories may take an image or a video as one.
 */
import { closestShape, sizeAspect } from './channel-shape'
import type {
  CreateSocialPostInput,
  SocialChannel,
  SocialClaimSource,
  SocialComposeProposal,
  SocialPost,
  SocialPostKind,
  SocialPostTargetInput,
  SocialPostTargetOptions,
  SocialPostVariable,
} from '@/lib/api-client'
import { browserTimezone, wallInZone, zonedWallToIso } from '@/lib/social-time'
import { defaultKind, draftTargets } from '../socials-composer-model'
import { lengthLabel } from './socials-calendar-model'

export type EditorFormat = 'image' | 'carousel' | 'video' | 'text'
export const EDITOR_FORMATS: ReadonlyArray<{ value: EditorFormat; label: string }> = [
  { value: 'image', label: 'Image' },
  { value: 'carousel', label: 'Carousel' },
  { value: 'video', label: 'Video' },
  { value: 'text', label: 'Text only' },
]

export const MIN_SLIDES = 2
export const MAX_SLIDES = 10
export const DEFAULT_SLIDES = 5
export const SLIDES_VARIABLE = 'slides'
/** Spoken words per second of video: the composer's budget (US-B103). */
export const WORDS_PER_SECOND = 2.5
export const NEW_POST_TITLE = 'Untitled post'

/** The media key of the person's own file as the whole post: an upload or a Library pick. */
export const OWN_FILE_ASPECT = 'original'

export interface EditorSlot {
  /** YYYY-MM-DD and HH:mm, wall time in `timezone`. */
  date: string
  time: string
  timezone: string
}

export interface EditorDraft {
  title: string
  brief: string
  format: string
  templateId: string | null
  lengthSeconds: number | null
  base: string
  perChannel: Record<string, string>
  variables: Record<string, SocialPostVariable>
  sources: Record<string, SocialClaimSource>
  /** Ticked channel (toolkit) → the post kind it publishes. */
  kinds: Record<string, SocialPostKind>
  options: Record<string, SocialPostTargetOptions>
  /** The AI footage switch for the hook and b-roll; null: as the post asks. */
  footageOn: boolean | null
  /** PRD-251B US-B305: the footage the post asks for, slot → prompt (an AI image's, or the b-roll's). */
  footage: Record<string, string>
  slot: EditorSlot | null
}

export function newDraft(): EditorDraft {
  return {
    title: '', brief: '', format: 'image', templateId: null, lengthSeconds: null, base: '', perChannel: {},
    variables: {}, sources: {}, kinds: {}, options: {}, footageOn: false, footage: {}, slot: null,
  }
}

function slotOfPost(post: SocialPost): EditorSlot | null {
  const at = post.planned_for ?? post.scheduled_for
  if (!at) return null
  const timezone = post.timezone || browserTimezone()
  const wall = wallInZone(at, timezone)
  return { date: wall.slice(0, 10), time: wall.slice(11, 16), timezone }
}

export function draftFromPost(post: SocialPost): EditorDraft {
  const targets = post.targets ?? []
  return {
    title: post.title,
    brief: post.brief ?? '',
    format: post.format ?? 'image',
    templateId: post.template_id,
    lengthSeconds: post.length_seconds,
    base: post.copy?.base ?? '',
    perChannel: { ...((post.copy?.channels as Record<string, string> | undefined) ?? {}) },
    variables: { ...(post.variables as Record<string, SocialPostVariable>) },
    sources: { ...(post.sources as Record<string, SocialClaimSource>) },
    kinds: Object.fromEntries(targets.map((t) => [t.toolkit, t.post_kind])),
    options: Object.fromEntries(targets.filter((t) => Object.keys(t.options ?? {}).length > 0).map((t) => [t.toolkit, t.options])),
    footageOn: null,
    footage: Object.fromEntries(Object.entries(post.footage ?? {}).map(([slot, asked]) => [slot, asked.prompt])),
    slot: slotOfPost(post),
  }
}

/** The template's slots the AI footage switch fills: its generatable ones that are not stills. */
export function videoSlotsOf(footageSlots: ReadonlyArray<string>, imageSlots: ReadonlyArray<string>): string[] {
  return footageSlots.filter((slot) => !imageSlots.includes(slot))
}

/** Whether the AI footage switch is on: as set, else whether the post asks for the hook or b-roll. */
export function footageSwitchOn(draft: EditorDraft, videoSlots: ReadonlyArray<string>): boolean {
  return draft.footageOn ?? videoSlots.some((slot) => slot in draft.footage)
}

/**
 * The footage a save asks for: the hook and b-roll while the switch is on (a video's, from
 * its brief), and every AI image slot the post asks for, with its own prompt, so a save
 * never drops the options made for it (PRD-251B US-B305). null when it asks for none.
 */
export function footageAsked(draft: EditorDraft, footageSlots: ReadonlyArray<string>, imageSlots: ReadonlyArray<string>) {
  const brief = draft.brief.trim() || draft.title.trim()
  const videoSlots = videoSlotsOf(footageSlots, imageSlots)
  const switched = draft.format === 'video' && footageSwitchOn(draft, videoSlots) ? videoSlots : []
  const stills = imageSlots.filter((slot) => footageSlots.includes(slot) && draft.footage[slot])
  const asked = Object.fromEntries([
    ...switched.map((slot) => [slot, { prompt: brief }] as const),
    ...stills.map((slot) => [slot, { prompt: draft.footage[slot] }] as const),
  ])
  return Object.keys(asked).length > 0 ? asked : null
}

/** The editor's fields as the post saves them (POST /posts or PATCH). */
export function postFields(draft: EditorDraft, footageSlots: ReadonlyArray<string>, imageSlots: ReadonlyArray<string> = []): CreateSocialPostInput {
  const video = draft.format === 'video'
  const footage = footageAsked(draft, footageSlots, imageSlots)
  return {
    title: draft.title.trim() || NEW_POST_TITLE,
    brief: draft.brief.trim() || null,
    copy: { base: draft.base, channels: { ...draft.perChannel } },
    format: draft.format,
    template_id: draft.format === 'text' ? null : draft.templateId,
    length_seconds: video ? draft.lengthSeconds : null,
    variables: draft.variables,
    sources: draft.sources,
    footage,
  }
}

export function editorTargets(draft: EditorDraft): SocialPostTargetInput[] {
  return draftTargets(draft)
}

/** The slot as PUT /slot takes it: an instant (ISO, UTC) and its zone; null clears it. */
export function slotInput(slot: EditorSlot | null): { plannedFor: string | null; timezone: string } {
  if (!slot || !slot.date || !slot.time) return { plannedFor: null, timezone: slot?.timezone || browserTimezone() }
  return { plannedFor: zonedWallToIso(`${slot.date}T${slot.time}`, slot.timezone), timezone: slot.timezone }
}

/** Whether the slot on screen differs from the post's: only then is PUT /slot called. */
export function slotChanged(post: SocialPost | null, slot: EditorSlot | null): boolean {
  const wanted = slotInput(slot).plannedFor
  const had = post?.planned_for ?? null
  const same = wanted === had || (wanted !== null && had !== null && new Date(wanted).getTime() === new Date(had).getTime())
  return !same || (slot !== null && (post?.timezone ?? null) !== slot.timezone)
}

/** The post's visual is the person's own file, with no template (3 Oct 2026): there is nothing
 * to render, so it goes for approval as it is (the server refuses to render it). */
export function isOwnFilePost(post: Pick<SocialPost, 'media'> | null, draft: Pick<EditorDraft, 'templateId'>): boolean {
  return !draft.templateId && !!post?.media && OWN_FILE_ASPECT in post.media
}

export function slidesOf(draft: Pick<EditorDraft, 'variables'>): number {
  const value = Number(draft.variables[SLIDES_VARIABLE]?.value)
  return Number.isFinite(value) && value >= MIN_SLIDES ? Math.min(value, MAX_SLIDES) : DEFAULT_SLIDES
}

export function withSlides(draft: EditorDraft, slides: number): EditorDraft {
  const clamped = Math.max(MIN_SLIDES, Math.min(MAX_SLIDES, slides))
  return { ...draft, variables: { ...draft.variables, [SLIDES_VARIABLE]: { value: clamped, claim: false } } }
}

/** "About 75 spoken words fit in 0:30." */
export function wordsHint(seconds: number): string {
  return `About ${Math.round(seconds * WORDS_PER_SECOND)} spoken words fit in ${lengthLabel(seconds)}.`
}

const VIDEO_KINDS: ReadonlyArray<SocialPostKind> = ['video', 'reel', 'short']
const VISUAL_KINDS: ReadonlyArray<SocialPostKind> = ['image', 'carousel']

function availableKinds(channel: SocialChannel): SocialPostKind[] {
  return channel.post_kinds.filter((kind) => kind.available).map((kind) => kind.kind)
}

/** Why `channel` cannot take a post of `format`, or null when it can. */
export function channelBlock(channel: SocialChannel, format: string): string | null {
  const kinds = availableKinds(channel)
  if (kinds.length === 0) return 'Not available'
  if (format === 'text') return kinds.includes('text' as SocialPostKind) ? null : 'Needs an image or video'
  if (format === 'video') return kinds.some((k) => VIDEO_KINDS.includes(k)) ? null : 'Takes images only'
  return kinds.some((k) => VISUAL_KINDS.includes(k)) ? null : 'Takes video only'
}

const STORY_FORMATS: ReadonlyArray<string> = ['image', 'video']
const STORY_KIND: SocialPostKind = 'story'

/** Whether `channel` can post a post of `format` as a story (PRD-251C, US-C301). */
export function canStory(channel: SocialChannel, format: string): boolean {
  return STORY_FORMATS.includes(format) && availableKinds(channel).includes(STORY_KIND)
}

/** The draft with a ticked channel posting as a story, or back as the format's own kind. */
export function withStory(draft: EditorDraft, channel: SocialChannel, on: boolean): EditorDraft {
  if (!(channel.toolkit in draft.kinds)) return draft
  const kind = on && canStory(channel, draft.format) ? STORY_KIND : defaultKind(channel, draft.format)
  return kind ? { ...draft, kinds: { ...draft.kinds, [channel.toolkit]: kind } } : draft
}

/** A ticked channel's kind in `format`: a story stays one where the format can be a story. */
function kindIn(channel: SocialChannel, format: string, current: SocialPostKind | undefined): SocialPostKind | null {
  return current === STORY_KIND && canStory(channel, format) ? STORY_KIND : defaultKind(channel, format)
}

/** The draft in `format`: a new kind of visual drops the template and the length, and
 * each ticked channel takes the format's kind (a story stays one), or is left out when it cannot. */
export function withFormat(draft: EditorDraft, format: string, channels: ReadonlyArray<SocialChannel>): EditorDraft {
  const kinds: Record<string, SocialPostKind> = {}
  for (const toolkit of Object.keys(draft.kinds)) {
    const channel = channels.find((c) => c.toolkit === toolkit)
    const kind = channel && !channelBlock(channel, format) ? kindIn(channel, format, draft.kinds[toolkit]) : null
    if (kind) kinds[toolkit] = kind
  }
  const sameKind = (format === 'video') === (draft.format === 'video')
  return {
    ...draft,
    format,
    kinds,
    templateId: sameKind && format !== 'text' ? draft.templateId : null,
    lengthSeconds: format === 'video' && sameKind ? draft.lengthSeconds : null,
  }
}

export function withChannelTicked(draft: EditorDraft, channel: SocialChannel, on: boolean): EditorDraft {
  const { [channel.toolkit]: _dropped, ...kinds } = draft.kinds
  if (!on) return { ...draft, kinds }
  const kind = defaultKind(channel, draft.format)
  if (!kind || channelBlock(channel, draft.format)) return draft
  return { ...draft, kinds: { ...kinds, [channel.toolkit]: kind } }
}

/** A post kind's aspect on a channel (a reel, a short, a story and TikTok are 9:16). */
const KIND_ASPECT: Record<string, string> = {
  'twitter:image': '16:9', 'twitter:video': '16:9',
  'linkedin:image': '1:1', 'linkedin:video': '16:9', 'linkedin:carousel': '4:5',
  'instagram:image': '1:1', 'instagram:carousel': '4:5', 'instagram:reel': '9:16', 'instagram:story': '9:16',
  'tiktok:video': '9:16', 'youtube:short': '9:16', 'youtube:video': '16:9',
}
const DEFAULT_ASPECT: Record<string, string> = { video: '16:9', reel: '9:16', short: '9:16', story: '9:16', carousel: '4:5', image: '1:1' }

export function aspectOf(toolkit: string, kind: string): string | null {
  return KIND_ASPECT[`${toolkit}:${kind}`] ?? DEFAULT_ASPECT[kind] ?? null
}

/** "1080 × 1920": the template's size the channel gets (its closest shape, channel-shape.ts),
 * else the channel's aspect itself. */
export function sizeFor(toolkit: string, kind: string, templateSizes: ReadonlyArray<string>): string {
  if (kind === 'text') return 'Text'
  const aspect = aspectOf(toolkit, kind)
  const size = closestShape(templateSizes, (s) => s, aspect)
  return size ? size.replace('x', ' × ') : aspect ?? kind
}

/** The distinct aspect ratios the ticked channels render in, in channel order. */
/** The ratios a render makes, one per size the ticked channels get (their closest of the
 * template's sizes); before a template is chosen, the channels' own aspects. */
export function renderRatios(draft: Pick<EditorDraft, 'kinds'>, templateSizes: ReadonlyArray<string> = []): string[] {
  const ratios = Object.entries(draft.kinds).map(([toolkit, kind]) => {
    const aspect = kind === 'text' ? null : aspectOf(toolkit, kind)
    const size = templateSizes.length ? closestShape(templateSizes, (s) => s, aspect) : null
    return size ? sizeAspect(size) : aspect
  })
  return Array.from(new Set(ratios.filter((r): r is string => !!r)))
}

/** Redraft with Auto: the proposal replaces the copy, the variables and the sources only. */
export function withProposal(draft: EditorDraft, proposal: SocialComposeProposal): EditorDraft {
  return {
    ...draft,
    base: proposal.copy.base,
    perChannel: { ...proposal.copy.per_channel },
    variables: { ...proposal.variables },
    sources: { ...proposal.sources },
  }
}
