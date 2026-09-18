'use client'

/**
 * MarkdownView — the one renderer for document-shaped markdown: file previews
 * (the Deliverables modal, the Workspace Explorer, the file widget), blog
 * drafts, skill bodies, mission results and report artifacts.
 *
 * PRD-244 review (Gerard, 2026-09-17): every one of those surfaces wrapped
 * react-markdown in `prose prose-sm dark:prose-invert`, but
 * `@tailwindcss/typography` is not a dependency of this app — those classes
 * have never done anything. Tailwind's preflight then strips heading sizes,
 * list markers and block margins, so a report rendered as a wall of text.
 * The typography lives in the `.md-view` block in globals.css, on the theme
 * tokens, so it follows the style (Classic / Studio) and the tone.
 *
 * Chat bubbles keep `chatMarkdownComponents` — conversation, not documents.
 */
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { CodeBlock } from '@/components/chatbot/code-block'
import { extractCodeChild } from '@/components/chatbot/markdown-components'
import { cn } from '@/lib/utils'

/** Fenced blocks get the copy-button card; everything else is styled by CSS. */
export const documentMarkdownComponents = {
  a: ({ href, children, ...props }: any) => (
    <a {...props} href={href} target="_blank" rel="noreferrer">
      {children}
    </a>
  ),
  pre: ({ children }: any) => {
    const extracted = extractCodeChild(children)
    if (!extracted) return <pre>{children}</pre>
    return <CodeBlock code={extracted.code} language={extracted.language} />
  },
  // GFM task lists render an input; it is a read-only marker, never a control.
  input: ({ checked, ...props }: any) => (
    <input {...props} type="checkbox" checked={!!checked} readOnly disabled className="md-task" />
  ),
}

export interface MarkdownViewProps {
  children: string | null | undefined
  /** `compact` for side panels and cards; `default` for full-page reads. */
  density?: 'default' | 'compact'
  /** Per-surface element overrides, merged over the document defaults. */
  components?: Record<string, unknown>
  /** Elements to drop (their children are kept), e.g. `['img']`. */
  disallowedElements?: string[]
  className?: string
}

export function MarkdownView({
  children,
  density = 'default',
  components,
  disallowedElements,
  className,
}: MarkdownViewProps) {
  return (
    <div className={cn('md-view', density === 'compact' && 'md-view-compact', className)}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{ ...documentMarkdownComponents, ...(components ?? {}) } as any}
        disallowedElements={disallowedElements}
        unwrapDisallowed={disallowedElements ? true : undefined}
      >
        {children ?? ''}
      </ReactMarkdown>
    </div>
  )
}
