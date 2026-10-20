'use client'

import type { ComponentPropsWithoutRef } from 'react'
import { MarkdownView } from '@automatos/theme-sdk'

import styles from './chat.module.css'

/**
 * PRD-260 §4 — Auto's answer as markdown, through the core's renderer
 * (the SDK's MarkdownView, PRD-261 A-14a: components/shared/markdown-view.tsx,
 * react-markdown 9 and remark-gfm, no rehype-raw). Automatos Play's rules for untrusted model output ride on top:
 * raw HTML in the text is dropped (skipHtml), never printed as literal tags;
 * links open in a new tab without handing it this window; an image shows as
 * its alt text, because an image URL is fetched with no click, so injected
 * instructions could use one to send chat content to any host; and a fenced
 * block is a plain <pre>, so Play's own pre rules style it (the core's
 * CodeBlock card is Studio's dark one).
 */

function ExternalLink({ href, title, children }: ComponentPropsWithoutRef<'a'>) {
  return (
    <a href={href} title={title} target="_blank" rel="noopener noreferrer">
      {children}
    </a>
  )
}

function ImageAsText({ alt }: ComponentPropsWithoutRef<'img'>) {
  return alt ? <span className={styles.imageAlt}>{alt}</span> : null
}

/** A fenced block printed as it is, in the surface's own pre rules (P260-FX-13, M36). */
function PrintedPre({ children }: ComponentPropsWithoutRef<'pre'>) {
  return <pre>{children}</pre>
}

/** Play's rules for untrusted markdown, also on a Deliverable's page (components/play/deliverables). */
export const PLAY_MARKDOWN_COMPONENTS = { a: ExternalLink, img: ImageAsText, pre: PrintedPre }

export function PlayMarkdown({ text }: { readonly text: string }) {
  return (
    <MarkdownView density="compact" components={PLAY_MARKDOWN_COMPONENTS} skipHtml className={styles.markdown}>
      {text}
    </MarkdownView>
  )
}
