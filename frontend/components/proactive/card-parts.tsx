'use client'

/**
 * PRD-268 §6.12.3, §6.12.4, K17 (US-C21c) — what the proactive default components share.
 *
 * Every class here is a token (no hex, no orange, no glass), so the same markup reads right in
 * Studio light, Studio dark and Classic: each style's token file decides the colour, and Studio's
 * radius override maps rounded-xl. The motion is one entrance per component (a fade plus a 4 px
 * rise over 180 ms); under reduced motion the rise is dropped and only the fade is left. Nothing loops.
 */
import type { ReactNode } from 'react'
import { aiLabelText } from '@/lib/ai-label/copy'
import { cn } from '@/lib/utils'

/** The component's one entrance. */
export const ENTRANCE_CLASS =
  'animate-in fade-in-0 slide-in-from-bottom-1 duration-[180ms] motion-reduce:slide-in-from-bottom-0'

/** A proactive card: flat paper, 1 px border. */
export const CARD_CLASS = 'rounded-xl border border-border bg-card p-4 text-card-foreground'

/** The view's one ink primary (Start). */
export const PRIMARY_BUTTON_CLASS =
  'inline-flex items-center gap-1 rounded-xl bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground ' +
  'transition-colors hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring ' +
  'disabled:pointer-events-none disabled:opacity-60'

/** Every other control: quiet, bordered. */
export const QUIET_BUTTON_CLASS =
  'inline-flex items-center gap-1 rounded-xl border border-border px-3 py-1.5 text-sm text-foreground ' +
  'transition-colors hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring ' +
  'disabled:pointer-events-none disabled:opacity-60'

/** A control drawn as text ("Start →", "Show 3 more"). */
export const TEXT_BUTTON_CLASS =
  'rounded-md text-sm font-medium text-foreground underline-offset-4 hover:underline ' +
  'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-60'

/** studio-eyebrow, with the same look spelled out for Classic (PageHeader's eyebrow). */
const EYEBROW_CLASS = 'studio-eyebrow font-mono text-[11px] font-semibold uppercase tracking-[0.08em] text-muted-foreground'

export function Eyebrow({ children, className }: { readonly children: ReactNode; readonly className?: string }) {
  return <p className={cn(EYEBROW_CLASS, className)}>{children}</p>
}

/**
 * The host's tiny "Auto is an AI" caption (Q29): the words of the in-app AI label
 * (lib/ai-label/copy.ts), in the muted-foreground token, never a pill. Shown once per card,
 * outside the fence of Auto's words.
 */
export function AiCaption({ className }: { readonly className?: string }) {
  const text = aiLabelText(null)
  return (
    <span role="note" aria-label={text} data-proactive-ai-caption="" className={cn('text-[11px] font-medium leading-4 text-muted-foreground', className)}>
      {text}
    </span>
  )
}
