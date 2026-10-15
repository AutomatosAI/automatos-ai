import { UNTITLED } from '@/components/play/chat/deliverables'
import { MADE_BY_SOURCE } from '@/components/workspace/gallery-view/made-by'
import type { Deliverable } from '@/hooks/use-deliverables-api'
import { deliverableFileUrl } from '@/lib/deliverables/download'

import { kindOf, shelfDate } from './shelf'

/**
 * PRD-260 §4, K8 — one Deliverable's page (Ember mock 05) as pure data, ported
 * from Automatos Play's lib/deliverables/detail.ts onto the core's Deliverable
 * (GET /api/deliverables/{id}?include_content=true): how its content shows, the
 * date and "Made by" lines, where it came from, what it was made from (its
 * sources), and the words "Ask Auto about this" puts in a new chat's message
 * box. "Now" is always passed in.
 */

/** File types read as markdown; any other text file shows verbatim in a code block. */
const MARKDOWN_TYPES: ReadonlySet<string> = new Set(['', 'md', 'markdown', 'mdx', 'txt', 'text'])
/**
 * Files the core serves as bytes (_BINARY_EXTENSIONS in services/deliverable_service.py).
 * Their inline content is the bytes read as text, so it is never shown as text.
 */
const BINARY_TYPES: ReadonlySet<string> = new Set([
  'png', 'jpg', 'jpeg', 'gif', 'webp', 'svg', 'ico', 'bmp', 'tif', 'tiff',
  'pdf', 'docx', 'xlsx', 'pptx', 'zip', 'tar', 'gz',
  'mp4', 'mp3', 'wav', 'ogg', 'webm', 'mov', 'mkv',
])
/** The core reads a report as markdown whatever its file says (DeliverablePreview). */
const MARKDOWN_KINDS: ReadonlySet<string> = new Set(['report', 'blog_post'])
const CODE_LANGUAGE = /^[a-z0-9]{1,16}$/
const MIN_FENCE = 3
/** Nouns for "Play can't preview this …" where the type word doesn't read as one. */
const KIND_NOUNS: Readonly<Record<string, string>> = { slide: 'slide deck', blog_post: 'blog post' }
/** Where a Deliverable came from (the core's source_type), in the side panel's words. */
const SOURCE_WORDS: Readonly<Record<string, string>> = {
  chat: 'A chat',
  task: 'A Task',
  mission: 'A Mission',
  heartbeat: 'A heartbeat',
  playbook: 'A Playbook',
  trigger: 'A trigger',
  upload: 'An upload',
  document: 'A template',
  social_post: 'A social post',
}
/** The longest title "Ask Auto about this" quotes. */
export const ASK_TITLE_CHARS = 120
const MADE_BY_DEFAULT = 'Auto'
/**
 * Sources Auto didn't make (the owner's uploads, documents made in Templates,
 * Socials posts), with the line under the title. "Made by" takes the core's own
 * word for them (made-by.ts, the Deliverables cards'): never "Auto".
 */
const MADE_ELSEWHERE_LINES: Readonly<Record<string, string>> = {
  upload: MADE_BY_SOURCE.upload,
  document: `Made in ${MADE_BY_SOURCE.document}`,
  social_post: `Made in ${MADE_BY_SOURCE.social_post}`,
}
/** The most sources the side panel lists (extra.sources is off the wire: any length). */
export const SOURCES_SHOWN_MAX = 20
const WEB_ADDRESS = /^https?:\/\//i
const WWW_PREFIX = /^www\./i

export type ContentView =
  /** Markdown (or a text file wrapped as a code block) for the core's markdown renderer. */
  | { readonly kind: 'text'; readonly markdown: string; readonly truncated: boolean }
  | { readonly kind: 'empty' }
  /** A picture or a video, fetched from `url` through the core's authenticated file path. */
  | { readonly kind: 'image' | 'video'; readonly url: string }
  /** Nothing to show here: the file card, with Download when `url` names the file. */
  | { readonly kind: 'file'; readonly reason: 'unsupported' | 'unreadable'; readonly url: string | null }

function hasOwn(record: Readonly<Record<string, string>>, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(record, key)
}

/** "report.md" → "md"; "" when the name has no extension. */
function extensionOf(name: string | null | undefined): string {
  const last = (name ?? '').split('/').pop() ?? ''
  const dot = last.lastIndexOf('.')
  return dot > 0 ? last.slice(dot + 1) : ''
}

/** The file's type as an extension, lowercase, without a dot: its file_type, else its name's. */
export function fileTypeOf(deliverable: Pick<Deliverable, 'file_type' | 'file_name' | 'file_path'>): string {
  const type = deliverable.file_type || extensionOf(deliverable.file_name || deliverable.file_path)
  return type.trim().toLowerCase().replace(/^\./, '')
}

/** The text in a code fence longer than any run of backticks inside it, so it can't close early. */
export function fenced(text: string, language: string): string {
  // reduce, not Math.max(...runs): a 1 MB file can hold more runs than a call takes arguments.
  const longest = (text.match(/`+/g) ?? []).reduce((most, run) => Math.max(most, run.length), 0)
  const fence = '`'.repeat(Math.max(MIN_FENCE, longest + 1))
  const info = CODE_LANGUAGE.test(language) ? language : ''
  return `${fence}${info}\n${text}\n${fence}`
}

function textView(content: string, asMarkdown: boolean, fileType: string, truncated: boolean): ContentView {
  if (!content.trim()) return { kind: 'empty' }
  return { kind: 'text', markdown: asMarkdown ? content : fenced(content, fileType), truncated }
}

/** How the Deliverable's content shows on its page. */
export function contentView(deliverable: Deliverable): ContentView {
  const url = deliverableFileUrl(deliverable)
  const kind = kindOf(deliverable)
  if (kind === 'image' || kind === 'video') {
    return url ? { kind, url } : { kind: 'file', reason: 'unsupported', url }
  }
  const fileType = fileTypeOf(deliverable)
  const binary = BINARY_TYPES.has(fileType)
  const content = typeof deliverable.content === 'string' ? deliverable.content : null
  if (content !== null && !binary) {
    const asMarkdown = MARKDOWN_KINDS.has(kind) || MARKDOWN_TYPES.has(fileType)
    return textView(content, asMarkdown, fileType, deliverable.content_truncated === true)
  }
  const unreadable = content === null && Boolean(deliverable.content_error) && !binary
  return { kind: 'file', reason: unreadable ? 'unreadable' : 'unsupported', url }
}

/** "Play can't preview this spreadsheet here." */
export function unsupportedMessage(kind: string): string {
  const noun = hasOwn(KIND_NOUNS, kind) ? KIND_NOUNS[kind] : kind.replace(/_/g, ' ')
  return `Play can't preview this ${noun} here.`
}

export function detailTitle(deliverable: Pick<Deliverable, 'title'>): string {
  return deliverable.title?.trim() || UNTITLED
}

function twoDigits(value: number): string {
  return String(value).padStart(2, '0')
}

/** "Today, 15:42", "Mon, 09:05", "4 Oct, 18:30"; "" when the date is missing or unreadable. */
export function detailDate(createdAt: string | null | undefined, now: Date): string {
  const day = shelfDate(createdAt, now)
  if (!day || !createdAt) return ''
  const date = new Date(createdAt)
  return `${day}, ${twoDigits(date.getHours())}:${twoDigits(date.getMinutes())}`
}

function agentOf(agentName: string | null | undefined): string | null {
  const agent = agentName?.trim()
  return agent && agent.toLowerCase() !== MADE_BY_DEFAULT.toLowerCase() ? agent : null
}

/** "upload", "document" or "social_post" (lowercase, trimmed): a Deliverable Auto didn't make; else null. */
function madeElsewhere(sourceType: string | null | undefined): string | null {
  const source = sourceType?.trim().toLowerCase() ?? ''
  return hasOwn(MADE_ELSEWHERE_LINES, source) ? source : null
}

/**
 * The "Made by" value in the side panel: the agent; else the core's word for
 * an upload, a Templates document or a Socials post ("Uploaded", "Templates",
 * "Socials"); else Auto (a chat, a Task, a Mission…).
 */
export function madeBy(agentName: string | null | undefined, sourceType?: string | null): string {
  const elsewhere = madeElsewhere(sourceType)
  return agentOf(agentName) ?? (elsewhere ? MADE_BY_SOURCE[elsewhere] : MADE_BY_DEFAULT)
}

/**
 * The line under the title: "Made by Auto with Researcher"; without an agent
 * "Uploaded", "Made in Templates" or "Made in Socials" for what Auto didn't
 * make, else "Made by Auto".
 */
export function madeLine(agentName: string | null | undefined, sourceType?: string | null): string {
  const agent = agentOf(agentName)
  if (agent) return `Made by Auto with ${agent}`
  const elsewhere = madeElsewhere(sourceType)
  return elsewhere ? MADE_ELSEWHERE_LINES[elsewhere] : 'Made by Auto'
}

/** Where it came from: "A chat", "A Mission"; an unknown source in its own words; "" for none. */
export function sourceLabel(sourceType: string | null | undefined): string {
  const source = sourceType?.trim().toLowerCase() ?? ''
  if (!source) return ''
  if (hasOwn(SOURCE_WORDS, source)) return SOURCE_WORDS[source]
  const words = source.replace(/_/g, ' ')
  return words.charAt(0).toUpperCase() + words.slice(1)
}

function isRecord(value: unknown): value is Readonly<Record<string, unknown>> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function nonBlank(value: unknown): string | null {
  return typeof value === 'string' && value.trim() ? value.trim() : null
}

/** One source as a short label: a URL's host without www, else its title, else the text itself; null for nothing. */
function sourceLabelOf(value: unknown): string | null {
  const raw = isRecord(value) ? (nonBlank(value.url) ?? nonBlank(value.title)) : nonBlank(value)
  if (!raw || !WEB_ADDRESS.test(raw)) return raw
  try {
    return new URL(raw).hostname.replace(WWW_PREFIX, '') || raw
  } catch {
    // Not a URL after all: the text is still a readable label.
    return raw
  }
}

/**
 * What the Deliverable was made from (its extra.sources, Automatos Play's
 * "Sources · N"), checked off the wire: each a string or { url | title },
 * trimmed, a web address shown as its host; anything else is skipped, and at
 * most SOURCES_SHOWN_MAX are listed. [] when it carries none.
 */
export function sourcesOf(deliverable: Pick<Deliverable, 'extra'>): readonly string[] {
  const sources: unknown = isRecord(deliverable.extra) ? deliverable.extra.sources : undefined
  if (!Array.isArray(sources)) return []
  return sources
    .map(sourceLabelOf)
    .filter((label): label is string => label !== null)
    .slice(0, SOURCES_SHOWN_MAX)
}

/** What "Ask Auto about this" puts in the new chat's message box; a very long title is cut, not the words around it. */
export function askAboutText(title: string): string {
  const chars = Array.from(title)
  const shown = chars.length > ASK_TITLE_CHARS ? `${chars.slice(0, ASK_TITLE_CHARS - 1).join('').trimEnd()}…` : title
  return `About my Deliverable "${shown}": `
}
