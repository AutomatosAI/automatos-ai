/**
 * PRD-261 A-12a — the SDK's content cards: `cards.FilePreview`, `cards.MarkdownView` and
 * `cards.MadeBy`.
 *
 * FilePreview is the host's file preview (components/widgets/FileWidget/FilePreview.tsx): it loads a
 * file the backend serves only with the caller's token, which a theme cannot send. MarkdownView is
 * the host's sanitising Markdown renderer (components/shared/markdown-view.tsx: react-markdown, no
 * raw HTML plugin), so a theme never renders untrusted Markdown itself. MadeBy is the line every
 * Deliverable card shows for who made it (components/workspace/gallery-view/made-by.ts).
 */
import type { ReactElement } from 'react'

import { MarkdownView as HostMarkdownView } from '@/components/shared/markdown-view'
import { FilePreview as HostFilePreview } from '@/components/widgets/FileWidget/FilePreview'
import { madeBy } from '@/components/workspace/gallery-view/made-by'

export type FilePreviewType =
  | 'text'
  | 'image'
  | 'pdf'
  | 'code'
  | 'binary'
  | 'html'
  | 'markdown'
  | 'csv'
  | 'video'
  | 'audio'
  | 'docx'
  | 'xlsx'
  | 'json'

export interface FilePreviewProps {
  /** Text content (HTML source, Markdown, code, CSV, JSON, plain text). */
  readonly content?: string
  /** The API path of a binary file (PDF, image, video, audio, docx, xlsx), fetched with the caller's token. */
  readonly url?: string
  /** The kind of preview; inferred from the file name when omitted. */
  readonly previewType?: FilePreviewType
  readonly filename?: string
  readonly className?: string
}

export interface MarkdownViewProps {
  readonly children: string | null | undefined
  /** `compact` for side panels and cards; `default` for full-page reads. */
  readonly density?: 'default' | 'compact'
  /** Element overrides (the theme's own `a`, `img`, `pre`…), merged over the host's. */
  readonly components?: Readonly<Record<string, unknown>>
  /** Elements to drop, keeping their children (`['img']`). */
  readonly disallowedElements?: readonly string[]
  /** Drop raw HTML in the text instead of printing it as literal tags. Set it for model output. */
  readonly skipHtml?: boolean
  readonly className?: string
}

export interface MadeByProps {
  /** The agent that made it, when one did. */
  readonly agentName?: string | null
  /** Where it came from (`upload`, `mission`, `social_post`…), named when no agent made it. */
  readonly sourceType?: string | null
  readonly className?: string
}

/** The host's preview of a file. */
export function FilePreview(props: FilePreviewProps): ReactElement {
  return <HostFilePreview {...props} />
}

/** The host's Markdown renderer. */
export function MarkdownView({ components, disallowedElements, children, ...view }: MarkdownViewProps): ReactElement {
  return (
    <HostMarkdownView
      {...view}
      components={components === undefined ? undefined : { ...components }}
      disallowedElements={disallowedElements === undefined ? undefined : [...disallowedElements]}
    >
      {children}
    </HostMarkdownView>
  )
}

/** Who made a Deliverable, as the host's cards say it. */
export function MadeBy({ agentName, sourceType, className }: MadeByProps): ReactElement {
  return <span className={className}>{madeBy(agentName, sourceType)}</span>
}
