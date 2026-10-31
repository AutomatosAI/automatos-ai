'use client'

/**
 * PRD-270 US-270-08a (§5.7) — the visitor's slim header: the Automatos logo, Sign in and Start free.
 *
 * Nothing workspace-scoped: no workspace card, notifications, credits or profile. Start free is
 * the view's one ink primary; Sign in is secondary.
 */
import Link from 'next/link'
import { Button } from '@/components/ui/button'

export const VISITOR_SIGN_IN = 'Sign in'
export const VISITOR_START_FREE = 'Start free'

export function VisitorHeader() {
  return (
    <header className="sticky top-0 z-30 border-b border-border bg-background/95 px-4 py-3 backdrop-blur md:px-6">
      <div className="mx-auto flex max-w-6xl items-center justify-between gap-3">
        <Link href="/marketplace" className="flex items-center gap-3" aria-label="Automatos Marketplace">
          <img src="/brand/automatos-mark-hi.png" alt="Automatos logo" className="h-7 w-auto" draggable={false} />
          <span className="hidden text-base font-semibold tracking-wide text-foreground sm:inline">AUTOMATOS A.I.</span>
        </Link>
        <nav aria-label="Account" className="flex items-center gap-2">
          <Button asChild variant="ghost" size="sm">
            <Link href="/sign-in">{VISITOR_SIGN_IN}</Link>
          </Button>
          <Button asChild size="sm">
            <Link href="/sign-up">{VISITOR_START_FREE}</Link>
          </Button>
        </nav>
      </div>
    </header>
  )
}
