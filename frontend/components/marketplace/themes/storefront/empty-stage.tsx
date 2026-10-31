'use client'

/**
 * PRD-261 — the storefront with nothing listed yet: the hero's stage, empty of devices, saying so
 * and inviting the person to make the first theme ("Make your own": the Design Kit and Submit a
 * theme package). Nothing is drawn that the server did not list. A signed-out visitor's storefront
 * offers no submit (the listing source's `canPublish`, PRD-270 K20), so it says to sign in for it.
 */
import { Palette } from 'lucide-react'

import { NO_LISTINGS } from '../listing-format'
import { useListingSource } from '../listing-source'
import { EMPTY_BODY, EMPTY_TITLE, MAKE_YOUR_OWN_VISITOR_BODY } from './copy'
import { MakeYourOwnActions } from './make-your-own'

export function EmptyStage() {
  const { canPublish } = useListingSource()
  return (
    <section
      aria-label={NO_LISTINGS}
      data-testid="storefront-empty"
      className="relative overflow-hidden rounded-3xl border border-dashed border-border bg-card"
    >
      <div className="absolute inset-0 bg-gradient-to-br from-primary/10 via-transparent to-accent/10" aria-hidden />
      <div className="relative flex flex-wrap items-center gap-10 px-6 py-12 sm:px-14 lg:px-20">
        <div className="flex min-w-0 flex-[1_1_320px] flex-col gap-4">
          <p className="font-mono text-xs uppercase tracking-[0.12em] text-muted-foreground">{NO_LISTINGS}</p>
          <h3 className="text-4xl font-bold leading-tight tracking-tight text-foreground sm:text-5xl">{EMPTY_TITLE}</h3>
          <p className="max-w-lg text-base leading-relaxed text-muted-foreground">
            {canPublish ? EMPTY_BODY : MAKE_YOUR_OWN_VISITOR_BODY}
          </p>
          <div className="max-w-xs">
            <MakeYourOwnActions />
          </div>
        </div>
        <div
          className="flex aspect-[16/10] min-w-0 flex-[1_1_320px] items-center justify-center rounded-xl border border-dashed border-border bg-background/60"
          aria-hidden
        >
          <Palette className="h-12 w-12 text-muted-foreground" />
        </div>
      </div>
    </section>
  )
}
