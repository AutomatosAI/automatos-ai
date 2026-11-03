'use client'

/**
 * PRD-270 US-270-08a, US-270-09a (§5.7, K21) — the visitor's slim header over the one Marketplace
 * page: the Automatos logo, the way back to Studio, Sign in and Start free.
 *
 * Visitors come here from the Studio product site, so the logo and a quiet "Studio" link both go back
 * to it (STUDIO_SITE_URL). Nothing workspace-scoped: no workspace card, notifications, credits or
 * profile. Start free is the view's one ink primary; Studio and Sign in are ghost. Sign in and Start
 * free remember the Marketplace page the visitor is on and carry it as `redirect_url`, so the person
 * comes back to it after sign-in or sign-up (lib/marketplace/return-after-sign-in.ts).
 */
import Link from 'next/link'
import { usePathname, useSearchParams } from 'next/navigation'
import { ArrowLeft } from 'lucide-react'

import { Button } from '@/components/ui/button'
import {
  MARKETPLACE_PATH,
  STUDIO_SITE_URL,
  remember,
  safeReturnPath,
  signInUrl,
  signUpUrl,
} from '@/lib/marketplace/return-after-sign-in'

export const VISITOR_SIGN_IN = 'Sign in'
export const VISITOR_START_FREE = 'Start free'
export const VISITOR_STUDIO = 'Studio'
export const VISITOR_LOGO_LABEL = 'Automatos Studio'

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
        <a href={STUDIO_SITE_URL} className="flex items-center gap-3" aria-label={VISITOR_LOGO_LABEL}>
          <img src="/brand/automatos-mark-hi.png" alt="Automatos logo" className="h-7 w-auto" draggable={false} />
          <span className="hidden text-base font-semibold tracking-wide text-foreground sm:inline">AUTOMATOS A.I.</span>
        </a>
        <nav aria-label="Account" className="flex items-center gap-2">
          <Button asChild variant="ghost" size="sm" className="text-muted-foreground">
            <a href={STUDIO_SITE_URL}>
              <ArrowLeft className="mr-1 h-4 w-4" aria-hidden />
              {VISITOR_STUDIO}
            </a>
          </Button>
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
