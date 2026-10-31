'use client'

/**
 * PRD-270 W2 US-270-14c (K30) — the pieces the disclosure's sections are made of (disclosure-sections.tsx,
 * disclosure-plugin.tsx): a titled section, a list of lines each with a short note, and a block that keeps
 * server text as written. Every text is rendered as text, never parsed as markdown or HTML.
 */
import type { ReactNode } from 'react'

import { Badge } from '@/components/ui/badge'

import { NONE } from './item-copy'

export function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="space-y-2" aria-label={title}>
      <h3 className="text-sm font-semibold text-foreground">{title}</h3>
      {children}
    </section>
  )
}

/** A short note beside a line: "needs a connection", "not on this install". */
export function Note({ text, warn = false }: { text: string; warn?: boolean }) {
  return (
    <Badge variant="outline" className={warn ? 'border-warning/30 text-xs text-warning' : 'text-xs text-muted-foreground'}>
      {text}
    </Badge>
  )
}

/** A line of a list, with its note. */
export function Line({ text, note, warn = false }: { text: string; note?: string | null; warn?: boolean }) {
  return (
    <li className="flex flex-wrap items-center gap-2 text-sm">
      <span className="min-w-0 break-words text-foreground">{text}</span>
      {note && <Note text={note} warn={warn} />}
    </li>
  )
}

/** A list of lines, or "None". */
export function Lines({ children, empty }: { children: ReactNode[]; empty?: string }) {
  if (children.length === 0) return <p className="text-sm text-muted-foreground">{empty ?? NONE}</p>
  return <ul className="space-y-1.5">{children}</ul>
}

/** Server text kept as written: wrapped, never parsed. */
export function TextBlock({ text }: { text: string }) {
  return (
    <pre className="max-h-64 overflow-y-auto whitespace-pre-wrap break-words rounded-xl border border-border/30 bg-secondary/30 p-3 font-sans text-sm text-foreground">
      {text}
    </pre>
  )
}
