'use client'

import { useEffect } from 'react'

import { PLAY_MARKDOWN_COMPONENTS } from '@/components/play/chat/play-markdown'
import { MarkdownView } from '@/components/shared/markdown-view'
import { useAuthenticatedBlobUrl } from '@/components/widgets/FileWidget/FilePreview'
import { useApiFileDownload } from '@/hooks/use-api-file-download'
import { DOWNLOAD_FAILED, fileDownloadUrl } from '@/lib/deliverables/download'
import { type ContentView, unsupportedMessage } from '@/lib/play/detail'

import styles from './deliverable-content.module.css'
import page from './deliverable-detail.module.css'

/**
 * PRD-260 §4 — a Deliverable's content on its page (Ember mock 05), ported
 * from Automatos Play's components/deliverables/DeliverableContent.tsx onto the
 * core's own renderers and file paths:
 * - text and markdown on the sheet through the core's MarkdownView with Play's
 *   rules for model output: raw HTML dropped (skipHtml), links open in a new
 *   tab without this window, images show as their alt text, a code block in
 *   Play's own pre rules (PrintedPre) on the sheet's colours;
 * - a picture or a video through useAuthenticatedBlobUrl, the authenticated
 *   path the core's FilePreview loads them by (a blob: URL, so an SVG's
 *   scripts never run in its <img>);
 * - anything else as Play's file card, with the core's download
 *   (useApiFileDownload: the API client's auth, never a plain link) of the
 *   file's bytes (fileDownloadUrl: never a text file's JSON content route).
 */

export const TRUNCATED = 'This is the first part: the file is too long to show here in full.'
export const EMPTY = 'This Deliverable is empty.'
export const UNREADABLE = "Automatos couldn't read this file."
export const PICTURE_LOADING = 'Loading the picture…'
export const PICTURE_FAILED = "The picture didn't load."
export const VIDEO_LOADING = 'Loading the video…'
export const VIDEO_FAILED = "The video didn't load."
export const DOWNLOAD_FILE = 'Download the file'
export const DOWNLOADING = 'Downloading…'

export interface DeliverableContentProps {
  readonly view: ContentView
  /** The Deliverable's type (kindOf), for the file card's words. */
  readonly kind: string
  readonly title: string
  /** The name its file saves under. */
  readonly filename: string
}

export function DeliverableContent({ view, kind, title, filename }: DeliverableContentProps) {
  if (view.kind === 'text') {
    return (
      <div className={styles.document}>
        <MarkdownView components={PLAY_MARKDOWN_COMPONENTS} skipHtml className={styles.prose}>
          {view.markdown}
        </MarkdownView>
        {view.truncated && <p className={styles.sheetNote}>{TRUNCATED}</p>}
      </div>
    )
  }
  if (view.kind === 'empty') return <p className={page.note}>{EMPTY}</p>
  if (view.kind === 'image' || view.kind === 'video') return <Media kind={view.kind} url={view.url} title={title} />
  const message = view.reason === 'unreadable' ? UNREADABLE : unsupportedMessage(kind)
  return <FileCard message={message} url={view.url} filename={filename} />
}

interface MediaProps {
  readonly kind: 'image' | 'video'
  readonly url: string
  readonly title: string
}

const MEDIA_WORDS = {
  image: { loading: PICTURE_LOADING, failed: PICTURE_FAILED },
  video: { loading: VIDEO_LOADING, failed: VIDEO_FAILED },
} as const

/** A picture or a video, its bytes fetched with the API client's auth. */
function Media({ kind, url, title }: MediaProps) {
  const { src, error } = useAuthenticatedBlobUrl(url)
  useEffect(() => {
    if (error) console.error(`[deliverable] the ${kind} at ${url} didn't load:`, error)
  }, [error, kind, url])
  if (error) {
    return (
      <p className={page.error} role="alert">
        <span aria-hidden="true">✕</span> {MEDIA_WORDS[kind].failed}
      </p>
    )
  }
  if (!src) {
    return (
      <p className={page.note} role="status">
        {MEDIA_WORDS[kind].loading}
      </p>
    )
  }
  if (kind === 'video') {
    return (
      <video src={src} controls preload="metadata" aria-label={title} className={styles.media}>
        <track kind="captions" />
      </video>
    )
  }
  // A blob: URL from the authenticated fetch, which next/image can't optimise.
  // eslint-disable-next-line @next/next/no-img-element
  return <img src={src} alt={title} className={styles.media} />
}

interface FileCardProps {
  readonly message: string
  readonly url: string | null
  readonly filename: string
}

/** Nothing to show here: what the file is, and Download when it has one. */
function FileCard({ message, url, filename }: FileCardProps) {
  const { download, downloading } = useApiFileDownload(DOWNLOAD_FAILED)
  const file = url ? fileDownloadUrl(url) : null
  return (
    <div className={styles.unavailable}>
      <p className={page.note}>{message}</p>
      {file && (
        <button
          type="button"
          className={page.secondary}
          onClick={() => void download(file, filename)}
          disabled={downloading}
        >
          {downloading ? DOWNLOADING : DOWNLOAD_FILE}
        </button>
      )}
    </div>
  )
}
