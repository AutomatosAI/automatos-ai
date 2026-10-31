'use client'

/**
 * PRD-270 US-270-08a, US-270-09a (§5.7, K21) — the visitor's slim header: the Automatos logo,
 * Sign in and Start free.
 *
 * Nothing workspace-scoped: no workspace card, notifications, credits or profile. Start free is
 * the view's one ink primary; Sign in is secondary. Both remember the Marketplace page the visitor
 * is on and carry it as `redirect_url`, so the person comes back to it after sign-in or sign-up
 * (lib/marketplace/return-after-sign-in.ts).
 */
import Link from 'next/link'
import { usePathname, useSearchParams } from 'next/navigation'

import { Button } from '@/components/ui/button'
import {
  MARKETPLACE_PATH,
  remember,
  safeReturnPath,
  signInUrl,
  signUpUrl,
} from '@/lib/marketplace/return-after-sign-in'

export const VISITOR_SIGN_IN = 'Sign in'
export const VISITOR_START_FREE = 'Start free'

/** The Marketplace page the visitor is on, as a safe return path ('/marketplace' when it isn't one). */
function useMarketplaceReturnPath(): string {
  const pathname = usePathname() ?? MARKETPLACE_PATH
  const query = useSearchParams()?.toString() ?? ''
  return safeReturnPath(query ? `${pathname}?${query}` : pathname) ?? MARKETPLACE_PATH
}

export function VisitorHeader() {
  const returnPath = useMarketplaceReturnPath()
  const rememberReturn = () => remember(returnPath)
  return (
    <header className="sticky top-0 z-30 border-b border-border bg-background/95 px-4 py-3 backdrop-blur md:px-6">
      <div className="mx-auto flex max-w-6xl items-center justify-between gap-3">
        <Link href={MARKETPLACE_PATH} className="flex items-center gap-3" aria-label="Automatos Marketplace">
          <img src="/brand/automatos-mark-hi.png" alt="Automatos logo" className="h-7 w-auto" draggable={false} />
          <span className="hidden text-base font-semibold tracking-wide text-foreground sm:inline">AUTOMATOS A.I.</span>
        </Link>
        <nav aria-label="Account" className="flex items-center gap-2">
          <Button asChild variant="ghost" size="sm">
            <Link href={signInUrl(returnPath)} onClick={rememberReturn}>
              {VISITOR_SIGN_IN}
            </Link>
          </Button>
          <Button asChild size="sm">
            <Link href={signUpUrl(returnPath)} onClick={rememberReturn}>
              {VISITOR_START_FREE}
            </Link>
          </Button>
        </nav>
      </div>
    </header>
  )
}
