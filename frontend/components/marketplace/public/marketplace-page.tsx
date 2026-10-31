'use client'

/**
 * PRD-270 US-270-08a (K19) — the Marketplace page's client switch.
 *
 *   local            → today's page (MainLayout + MarketplaceHomepage); no Clerk hook runs.
 *   SaaS, signed in  → today's page, exactly.
 *   SaaS, signed out → the visitor's read-only storefront: no app shell, nothing workspace-scoped.
 *   SaaS, loading    → a quiet line, never the member page.
 *
 * isSaaS is a build-time constant, so the branch never changes which hooks a render calls.
 */
import { MainLayout } from '@/components/layout/main-layout'
import { MarketplaceHomepage } from '@/components/marketplace/marketplace-homepage'
import { usePageAPI } from '@/hooks/use-page-api'
import { useAuth } from '@/lib/auth-hooks'
import { isSaaS } from '@/lib/auth-edition'
import { VisitorMarketplace } from './visitor-marketplace'

export const MARKETPLACE_LOADING = 'Loading the Marketplace…'

function MemberMarketplace() {
  return (
    <MainLayout>
      <MarketplaceHomepage />
    </MainLayout>
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
  return isSignedIn ? <MemberMarketplace /> : <VisitorMarketplace />
}

/** The /marketplace page: today's page for members and the local edition, the visitor view signed out on SaaS. */
export function MarketplacePage() {
  usePageAPI('marketplace')
  return isSaaS ? <SaasMarketplace /> : <MemberMarketplace />
}
