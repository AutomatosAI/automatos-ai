'use client'

import type { ReactNode } from 'react'
import { isValidElement } from 'react'
import { CodeBlock } from './code-block'

/**
 * Shared ReactMarkdown component map for chat surfaces.
 *
 * react-markdown v9 removed the `inline` prop from the `code` renderer, so
 * inline-vs-block must be split structurally instead: fenced blocks always
 * arrive wrapped in a `pre` element, inline code never does. The `pre`
 * renderer therefore owns block rendering (it reads the child code element's
 * language + text and delegates to CodeBlock), and the `code` renderer only
 * ever styles inline spans. Destructuring `inline` (always undefined in v9)
 * was sending every single backtick — including table-cell values — through
 * the full CodeBlock card with header and Copy button.
 */

interface CodeNodeProps {
  className?: string
  children?: ReactNode
}

/** Pull the code text + language off the `code` element nested in a `pre`. */
function extractCodeChild(children: ReactNode): { code: string; language: string } | null {
  const nodes = Array.isArray(children) ? children : [children]
  const codeEl = nodes.find((c) => isValidElement<CodeNodeProps>(c))
  if (!codeEl || !isValidElement<CodeNodeProps>(codeEl)) return null
  const match = /language-(\w+)/.exec(codeEl.props.className || '')
  const raw = codeEl.props.children
  const code = String(Array.isArray(raw) ? raw.join('') : raw ?? '').replace(/\n$/, '')
  return { code, language: match?.[1] || '' }
}

export const chatMarkdownComponents = {
  p: ({ children }: any) => (
    <p className="text-foreground leading-relaxed tracking-[0.01em] dark:text-gray-100">{children}</p>
  ),
  strong: ({ children }: any) => (
    <strong className="text-foreground font-semibold dark:text-gray-100">{children}</strong>
  ),
  em: ({ children }: any) => <em className="text-muted-foreground italic dark:text-foreground/90">{children}</em>,
  a: ({ href, children }: any) => (
    <a
      href={href}
      target="_blank"
      rel="noreferrer"
      className="text-primary hover:text-primary/80 underline"
    >
      {children}
    </a>
  ),
  ul: ({ children }: any) => (
    <ul className="list-disc pl-5 space-y-1.5 text-foreground dark:text-gray-100">{children}</ul>
  ),
  ol: ({ children }: any) => (
    <ol className="list-decimal pl-5 space-y-1.5 text-foreground dark:text-gray-100">{children}</ol>
  ),
  li: ({ children }: any) => (
    <li className="text-foreground dark:text-gray-100 pl-1">{children}</li>
  ),
  // Inline code ONLY — block code never reaches this because `pre` below
  // renders fenced blocks itself via CodeBlock.
  code: ({ children }: any) => (
    <code className="rounded bg-primary/10 px-1.5 py-0.5 text-[13px] font-mono text-primary border border-primary/10">
      {children}
    </code>
  ),
  pre: ({ children }: any) => {
    const extracted = extractCodeChild(children)
    if (!extracted) {
      return <pre className="overflow-x-auto text-[13px]">{children}</pre>
    }
    return <CodeBlock code={extracted.code} language={extracted.language} />
  },
  blockquote: ({ children }: any) => (
    <blockquote className="border-l-2 border-primary/40 pl-4 py-1 bg-primary/5 rounded-r-lg text-foreground/80 dark:text-foreground/90 italic">
      {children}
    </blockquote>
  ),
  h1: ({ children }: any) => (
    <h1 className="text-lg font-semibold text-foreground dark:text-gray-100 pb-1 border-b border-border/30 mb-2">{children}</h1>
  ),
  h2: ({ children }: any) => (
    <h2 className="text-base font-semibold text-foreground dark:text-gray-100 pb-1 border-b border-border/20 mb-2">{children}</h2>
  ),
  h3: ({ children }: any) => (
    <h3 className="text-sm font-semibold text-foreground dark:text-gray-100 mb-1">{children}</h3>
  ),
  hr: () => (
    <hr className="border-0 h-px bg-gradient-to-r from-transparent via-primary/30 to-transparent my-4" />
  ),
  table: ({ children }: any) => (
    <div className="overflow-x-auto rounded-xl border border-border/60 bg-card/50 dark:border-gray-800/60 dark:bg-background/40">
      <table className="min-w-full divide-y divide-border/60 text-sm text-foreground dark:divide-gray-800/70 dark:text-gray-100">
        {children}
      </table>
    </div>
  ),
  thead: ({ children }: any) => (
    <thead className="bg-secondary/40 text-xs uppercase tracking-wide text-muted-foreground dark:bg-background/60 dark:text-muted-foreground">
      {children}
    </thead>
  ),
  tbody: ({ children }: any) => (
    <tbody className="divide-y divide-border/50 dark:divide-gray-800/70">{children}</tbody>
  ),
  tr: ({ children }: any) => (
    <tr className="hover:bg-secondary/40 transition-colors dark:hover:bg-background/60">{children}</tr>
  ),
  th: ({ children }: any) => (
    <th className="px-4 py-3 text-left font-semibold text-foreground/80 dark:text-foreground/90">
      {children}
    </th>
  ),
  td: ({ children }: any) => (
    <td className="px-4 py-3 align-top text-foreground dark:text-gray-200">{children}</td>
  ),
  // Images handled by ImageGallery — strip from markdown
  img: () => null,
}
