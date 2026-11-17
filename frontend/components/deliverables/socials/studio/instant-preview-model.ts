/**
 * PRD-251D US-D201 (E8) — the instant preview of an image post (pure): which posts show it,
 * what the editor asks the server for (the words on screen, the size the active channel gets,
 * whether a ticked channel posts it as a story, the photos it holds), and the page with the
 * kit's font faces and each photo put where the server marked them.
 *
 * The server fills and escapes every word (orchestrator modules/socials/instant_preview.py);
 * this file only places the workspace's own files at the page's markers. A page is shown only
 * when it opens with its own content security policy (`hasPagePolicy`), in a frame with
 * sandbox="allow-scripts" and no same origin (instant-preview-frame.tsx). It is labelled a
 * preview: the file a person approves is always the render.
 */
import type { SocialChannel, SocialPost, SocialPreviewPageInput, SocialTemplateSummary } from '@/lib/api-client'
import type { PreviewPhotoFile } from '@/hooks/use-socials-instant-preview'
import { closestShape } from './channel-shape'
import { aspectOf, type EditorDraft } from './editor-model'
import { lookToSave } from './look-options-model'
import { previewStale } from './preview-model'

/** How long after the last change the page is asked for. */
export const INSTANT_PREVIEW_DELAY_MS = 600
export const INSTANT_PREVIEW_LABEL = 'Preview'
export const INSTANT_PREVIEW_TITLE = 'Instant preview of the post'
export const INSTANT_PREVIEW_NOTE =
  'Instant preview of the words on screen. Auto writes any field left empty when the post is made, and the file that is approved is always the render.'
export const PREVIEW_PAUSED = 'The preview waits:'
/** The policy the server's page opens with (modules/socials/instant_preview.py POLICY_META). */
export const PAGE_POLICY_META =
  `<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data: blob:; font-src data:">`
/** Where the kit's font faces go, and where a photo spot's picture goes (the server's markers). */
export const FONTS_MARKER = '/*&automatos-preview-fonts;*/'
export const photoMarker = (slot: string): string => `&automatos-preview-photo-${slot};`

const IMAGE_FORMAT = 'image'
const STORY_KIND = 'story'
const DONE = 'done'
// A photo goes into a quoted src only as a data: URI of an image, nothing that could end the quote.
const SAFE_PHOTO = /^data:image\/(png|jpeg|webp);base64,[A-Za-z0-9+/]+=*$/
const STYLE_CLOSE = /<\/style/i
const SCRIPT_OPEN = /<script/i
const SIZE = /^(\d{2,4})x(\d{2,4})$/

/** Whether the post's rendered file shows what is on screen: it has one, the server says it was
 * made from what the post holds now (`file_current`, US-D108: a saved edit after the render
 * leaves it older than the words), and nothing a render reads has changed since the post was saved. */
export function renderIsCurrent(post: SocialPost | null, draft: EditorDraft): boolean {
  return !!post && Object.keys(post.media ?? {}).length > 0 && post.file_current !== false && !previewStale(post, draft)
}

/** Whether the preview shows the instant page: an image post with a template whose file is not
 * current. A video, a carousel, a text post, a post with no template (left to "Let Auto pick", or
 * the person's own file, which never has one) and a current render keep the preview they have. */
export function wantsInstantPreview(draft: Pick<EditorDraft, 'format' | 'templateId'>, fileCurrent: boolean): boolean {
  return draft.format === IMAGE_FORMAT && !!draft.templateId && !fileCurrent
}

/** The pictures the post holds in the template's photo spots: each spot's chosen file. */
export function photoFilesOf(post: SocialPost | null, template: SocialTemplateSummary | null): PreviewPhotoFile[] {
  return (template?.image_slots ?? []).flatMap((slot) => {
    const held = post?.footage?.[slot]
    return held?.status === DONE && held.name ? [{ slot, name: held.name }] : []
  })
}

/** The photos that may go into the page (each a data: URI of an image), by spot. */
export function safePhotos(photos: Record<string, string> | undefined): Record<string, string> {
  return Object.fromEntries(Object.entries(photos ?? {}).filter(([, uri]) => SAFE_PHOTO.test(uri)))
}

function sendable(value: unknown): value is string | number | boolean {
  return typeof value === 'string' || typeof value === 'boolean' || (typeof value === 'number' && Number.isFinite(value))
}

/** The fields' values on screen: text, a finite number or a switch; anything else is left out (empty). */
export function valuesOf(draft: Pick<EditorDraft, 'variables'>): Record<string, string | number | boolean> {
  return Object.fromEntries(
    Object.entries(draft.variables).flatMap(([name, spec]) => {
      const value: unknown = spec?.value
      return sendable(value) ? [[name, value] as const] : []
    }),
  )
}

/** The template's size the active channel gets (its closest shape, as the render picks it), else its first. */
export function previewSize(templateSizes: ReadonlyArray<string>, draft: EditorDraft, active: SocialChannel | null): string | null {
  const aspect = active ? aspectOf(active.toolkit, draft.kinds[active.toolkit]) : null
  return closestShape(templateSizes, (size) => size, aspect) ?? templateSizes[0] ?? null
}

/** What the editor asks the page for. A story when any ticked channel posts it as one, as the render decides. */
export function previewInput(
  draft: EditorDraft, templateSizes: ReadonlyArray<string>, active: SocialChannel | null, photos: ReadonlyArray<string>,
): SocialPreviewPageInput {
  const look = lookToSave(draft.look ?? {})  // US-D301: null when every choice is the kit's own
  return {
    values: valuesOf(draft),
    size: previewSize(templateSizes, draft, active),
    story: Object.values(draft.kinds).includes(STORY_KIND),
    photos: [...photos].sort(),
    ...(look ? { look } : {}),
  }
}

/** Whether the page opens with its own content security policy, before any script of its. */
export function hasPagePolicy(page: string): boolean {
  const at = page.indexOf(PAGE_POLICY_META)
  const script = page.search(SCRIPT_OPEN)
  return at >= 0 && (script < 0 || at < script)
}

/** The page with the kit's font faces at the font marker and each photo at its spot's marker. */
export function pageWithFiles(page: string, fontsCss: string, photos: Record<string, string>): string {
  const fonts = STYLE_CLOSE.test(fontsCss) ? '' : fontsCss
  const withFonts = page.split(FONTS_MARKER).join(fonts)
  return Object.entries(safePhotos(photos)).reduce((html, [slot, uri]) => html.split(photoMarker(slot)).join(uri), withFonts)
}

/** "1080x1350" → { width: 1080, height: 1350 }; null for anything else. */
export function sizeOf(size: string | null): { width: number; height: number } | null {
  const match = SIZE.exec(size ?? '')
  return match ? { width: Number(match[1]), height: Number(match[2]) } : null
}
