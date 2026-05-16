'use client'

import type { ReactNode } from 'react'
import { motion } from 'framer-motion'
import { cn } from '@/lib/utils'
import { useIsStudio } from '@/hooks/use-studio-theme'

export interface PageHeaderProps {
  /** First word(s) of the title — rendered in foreground */
  title: string
  /** Accent word(s) of the title — orange gradient under classic theme,
   *  plain ink under Studio (the gradient-text utility is neutralised). */
  titleAccent: string
  /** Short description below the title. Renders as muted body copy.
   *  Use `lede` instead for the longer-form editorial paragraph under Studio. */
  subtitle?: string
  /** Optional mono uppercase eyebrow above the title.
   *  Studio editorial-first pattern (PRD §1). Skipped if absent. */
  eyebrow?: string
  /** Optional longer-form opening paragraph (1–3 sentences). Replaces
   *  `subtitle` styling with the studio-lede treatment. Studio editorial-first
   *  pattern (PRD §1). Falls through to `subtitle` styling under non-Studio. */
  lede?: ReactNode
  /** Right-aligned action group (typically buttons). */
  actions?: ReactNode
  className?: string
}

/**
 * PageHeader — the editorial-first page lede.
 *
 * Classic theme: two-word title with orange gradient on the accent, optional
 * subtitle, optional actions. Existing API preserved.
 *
 * Studio theme: serif headline (h1 picks up serif via globals.css `.studio`
 * scope automatically), optional mono uppercase eyebrow above, optional lede
 * paragraph below. The PRD §1 editorial-first principle lives here.
 */
export function PageHeader({
  title,
  titleAccent,
  subtitle,
  eyebrow,
  lede,
  actions,
  className,
}: PageHeaderProps) {
  const isStudio = useIsStudio()

  // Studio: editorial cc-* primitives. Matches Command Centre /
  // Assignments / Mission Detail headers. Title halves concatenate
  // (Studio neutralises the gradient on titleAccent anyway).
  if (isStudio) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.35 }}
        className={cn('cc-headrow', className)}
      >
        <div className="cc-head">
          {eyebrow && <p className="cc-eyebrow">{eyebrow}</p>}
          <h1 data-testid="page-title" className="cc-h1">
            {title}
            {titleAccent ? ` ${titleAccent}` : null}
          </h1>
          {lede ? (
            <p className="cc-sub">{lede}</p>
          ) : subtitle ? (
            <p className="cc-sub">{subtitle}</p>
          ) : null}
        </div>
        {actions && <div className="cc-actions">{actions}</div>}
      </motion.div>
    )
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className={cn(
        'flex flex-col sm:flex-row justify-between items-start gap-3',
        className
      )}
    >
      <div className="min-w-0">
        {eyebrow && (
          <p
            className="studio-eyebrow mb-1 text-[11px] uppercase tracking-[0.08em] text-muted-foreground font-mono font-semibold"
          >
            {eyebrow}
          </p>
        )}
        <h1
          data-testid="page-title"
          className="text-2xl md:text-3xl font-bold mb-1 md:mb-2"
        >
          {title} <span className="gradient-text">{titleAccent}</span>
        </h1>
        {lede ? (
          <p className="studio-lede max-w-[70ch] text-sm md:text-base text-muted-foreground leading-relaxed">
            {lede}
          </p>
        ) : subtitle ? (
          <p className="text-sm md:text-base text-muted-foreground">{subtitle}</p>
        ) : null}
      </div>
      {actions && (
        <div className="flex items-center gap-2 md:gap-3 shrink-0 flex-wrap">
          {actions}
        </div>
      )}
    </motion.div>
  )
}
