'use client'

/**
 * PRD-251D US-D201 (E8) — the instant preview the editor's Preview shows for an image post with a
 * template (instant-preview-model.ts says which posts): the page for what is on screen, asked for
 * a moment after the last change, with the kit's font faces and the post's photos put in it.
 *
 * No page while the post keeps today's preview, until a first page and the kit's fonts arrive, and
 * after any failure but a refusal. After a refusal (a field over its limit, say) the last page of
 * the same template stays, and the reason is said whether or not a page is on screen yet. A page is
 * shown only when it opens with its own policy.
 */
import { useEffect, useMemo, useState } from 'react'
import type { UseQueryResult } from '@tanstack/react-query'

import type { SocialChannel, SocialPost, SocialPreviewPage, SocialPreviewPageInput, SocialTemplateSummary } from '@/lib/api-client'
import { useSocialTemplates } from '@/hooks/use-socials-editor'
import { useSocialPreviewFonts, useSocialPreviewPage, useSocialPreviewPhotos } from '@/hooks/use-socials-instant-preview'
import { useDebouncedValue } from '@/components/workspace/gallery-view/use-debounced-value'
import type { EditorDraft } from './editor-model'
import {
  INSTANT_PREVIEW_DELAY_MS, hasPagePolicy, pageWithFiles, photoFilesOf, previewInput, safePhotos, sizeOf, wantsInstantPreview,
} from './instant-preview-model'

const HTTP_UNPROCESSABLE = 422

export interface InstantPreview {
  /** The page, the kit's fonts and the post's photos in place. */
  page: string
  /** The template's size it is drawn at. */
  width: number
  height: number
}

export interface InstantPreviewState {
  /** The page shown in place of the file; null keeps today's frame. */
  instant: InstantPreview | null
  /** Why the words on screen have no page of their own (the server's reason for refusing them);
   * null when the page shows them, or nothing was refused. */
  problem: string | null
}

interface InstantPreviewInput {
  post: SocialPost | null
  draft: EditorDraft
  templateSizes: ReadonlyArray<string>
  /** The Preview's channel tab: the size its channel gets. */
  active: SocialChannel | null
  /** The post's rendered file shows what is on screen (instant-preview-model `renderIsCurrent`). */
  fileCurrent: boolean
}

interface ShownPage {
  templateId: string
  input: SocialPreviewPageInput
  page: SocialPreviewPage
}

function useChosenTemplate(draft: EditorDraft): SocialTemplateSummary | null {
  const templates = useSocialTemplates(draft.format)
  return templates.data?.find((template) => template.id === draft.templateId) ?? null
}

/** What is on screen once it has stopped changing for a moment, and whether it has: `settled` is
 * false while the person is still typing. Compared as text, so an equal request made again on a
 * later render never restarts the wait. */
function useSettledInput(input: SocialPreviewPageInput): { input: SocialPreviewPageInput; settled: boolean } {
  const current = JSON.stringify(input)
  const latest = useDebouncedValue(current, INSTANT_PREVIEW_DELAY_MS)
  const parsed = useMemo(() => JSON.parse(latest) as SocialPreviewPageInput, [latest])
  return { input: parsed, settled: latest === current }
}

/** The newest page that answered its own request, with that request. */
function useShownPage(templateId: string | null, input: SocialPreviewPageInput, query: UseQueryResult<SocialPreviewPage>): ShownPage | null {
  const [shown, setShown] = useState<ShownPage | null>(null)
  const { data } = query
  useEffect(() => {
    if (templateId && data) setShown({ templateId, input, page: data })
  }, [templateId, input, data])
  return shown
}

/** The server's reason when it refused the newest words (422 with a sentence); null otherwise. */
function refusalOf(query: UseQueryResult<SocialPreviewPage>): string | null {
  const error = query.error as { status?: number; body?: { detail?: unknown } } | null
  const detail = error?.status === HTTP_UNPROCESSABLE ? error.body?.detail : undefined
  return query.isError && typeof detail === 'string' ? detail : null
}

export function useInstantPreview({ post, draft, templateSizes, active, fileCurrent }: InstantPreviewInput): InstantPreviewState {
  const wanted = wantsInstantPreview(draft, fileCurrent)
  const template = useChosenTemplate(draft)
  const files = useMemo(() => (wanted ? photoFilesOf(post, template) : []), [wanted, post, template])
  const photos = useSocialPreviewPhotos(post?.id ?? null, files)
  const held = useMemo(() => safePhotos(photos.data), [photos.data])
  const { input, settled } = useSettledInput(previewInput(draft, templateSizes, active, Object.keys(held)))
  // Asked for once the typing has stopped, the post's photos are read and the size is known; the
  // last page stays on screen meanwhile.
  const asking = wanted && settled && !photos.isInitialLoading && !!input.size ? draft.templateId : null
  const query = useSocialPreviewPage(asking, input)
  const shown = useShownPage(asking, input, query)
  const fonts = useSocialPreviewFonts(shown?.page.fonts_version ?? null)
  const page = useMemo(() => (shown ? pageWithFiles(shown.page.html, fonts.data?.css ?? '', held) : null), [shown, fonts.data, held])
  const size = sizeOf(shown?.input.size ?? null)
  const problem = wanted ? refusalOf(query) : null
  const failed = query.isError && problem === null
  const usable = wanted && !failed && shown?.templateId === draft.templateId && !fonts.isInitialLoading
  const instant = usable && page && size && hasPagePolicy(page) ? { page, ...size } : null
  return { instant, problem }
}
