'use client'

/**
 * PRD-270 (§5.7, one Marketplace page for everyone) — the /marketplace page. It is the same page,
 * MarketplaceHomepage, for everyone; what changes is the frame around it and who may add:
 *
 *   local            → MainLayout + MarketplaceHomepage; no Clerk hook runs.
 *   SaaS, signed in  → the same, exactly.
 *   SaaS, signed out → the slim visitor header (logo and Studio, Sign in, Start free) over the same
 *                      homepage, with the visitor's access (VisitorAccessProvider): it reads the
 *                      public routes, draws nothing workspace-scoped, and every add goes to sign-in.
 *   SaaS, loading    → a quiet line, never the page.
 *
 * isSaaS is a build-time constant, so the branch never changes which hooks a render calls.
 */
import { MainLayout } from '@/components/layout/main-layout'
import { MarketplaceHomepage } from '@/components/marketplace/marketplace-homepage'
import { usePageAPI } from '@/hooks/use-page-api'
import { useAuth } from '@/lib/auth-hooks'
import { isSaaS } from '@/lib/auth-edition'

import { VisitorAccessProvider } from '../access/visitor-access'
import { VisitorHeader } from './visitor-header'

export const MARKETPLACE_LOADING = 'Loading the Marketplace…'

function MemberMarketplace() {
  return (
    <MainLayout>
      <MarketplaceHomepage />
    </MainLayout>
  )
}

function SignedOutMarketplace() {
  return (
    <VisitorAccessProvider>
      <div className="min-h-screen bg-background text-foreground">
        <VisitorHeader />
        <main className="mx-auto max-w-6xl px-4 py-8 md:px-6">
          <MarketplaceHomepage />
        </main>
      </div>
    </VisitorAccessProvider>
  )
}

function SaasMarketplace() {
  const { isLoaded, isSignedIn } = useAuth()
  if (!isLoaded) {
    return (
      <p role="status" className="px-4 py-10 text-center text-sm text-muted-foreground">
        {MARKETPLACE_LOADING}
      </p>
    )
  }
  return isSignedIn ? <MemberMarketplace /> : <SignedOutMarketplace />
}

/** The /marketplace page: the one homepage, in the app shell or under the visitor header. */
export function MarketplacePage() {
  usePageAPI('marketplace')
  return isSaaS ? <SaasMarketplace /> : <MemberMarketplace />
}
