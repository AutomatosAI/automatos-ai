/**
 * PRD-270 US-270-08a — /marketplace, a server component.
 *
 * Public on SaaS (the middleware lists the exact path); the local edition serves every route.
 * The client switch decides what renders: a member sees today's page, a signed-out visitor the
 * read-only storefront.
 *
 * US-270-10 (K23): its metadata is a Marketplace link's share card, read server-side from the
 * public routes (lib/marketplace/share-card-read.ts), with the item's one canonical URL.
 */
import type { Metadata } from 'next'
import { headers } from 'next/headers'

import { MarketplacePage } from '@/components/marketplace/public/marketplace-page'
import { marketplaceMetadata } from '@/lib/marketplace/share-card-read'
import type { PageSearchParams } from '@/lib/marketplace/share-card-read'

interface MarketplacePageProps {
  searchParams: Promise<PageSearchParams>
}

export async function generateMetadata({ searchParams }: MarketplacePageProps): Promise<Metadata> {
  return marketplaceMetadata(await searchParams, await headers())
}

export default function Page() {
  return <MarketplacePage />
}
