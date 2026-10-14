'use client'

import type { ComponentPropsWithoutRef } from 'react'

import { MarkdownView } from '@/components/shared/markdown-view'

import styles from './chat.module.css'

/**
 * PRD-260 §4 — Auto's answer as markdown, through the core's renderer
 * (components/shared/markdown-view.tsx: react-markdown 9 and remark-gfm, no
 * rehype-raw, so raw HTML in the text never renders as HTML). Two of Automatos
 * Play's rules for untrusted model output ride on top as element overrides:
 * links open in a new tab without handing it this window, and an image shows as
 * its alt text, because an image URL is fetched with no click, so injected
 * instructions could use one to send chat content to any host.
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

const PLAY_MARKDOWN_COMPONENTS = { a: ExternalLink, img: ImageAsText }

export function PlayMarkdown({ text }: { readonly text: string }) {
  return (
    <MarkdownView density="compact" components={PLAY_MARKDOWN_COMPONENTS} className={styles.markdown}>
      {text}
    </MarkdownView>
  )
}
