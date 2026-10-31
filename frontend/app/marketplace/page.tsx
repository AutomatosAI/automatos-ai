/**
 * PRD-270 US-270-08a — /marketplace, a server component (US-270-10 adds its metadata here).
 *
 * Public on SaaS (the middleware lists the exact path); the local edition serves every route.
 * The client switch decides what renders: a member sees today's page, a signed-out visitor the
 * read-only storefront.
 */
import { MarketplacePage } from '@/components/marketplace/public/marketplace-page'

export default function Page() {
  return <MarketplacePage />
}
