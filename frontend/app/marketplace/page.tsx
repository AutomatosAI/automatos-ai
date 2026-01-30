'use client'

import { MainLayout } from '@/components/layout/main-layout'
import { MarketplaceHomepage } from '@/components/marketplace/marketplace-homepage'
import { usePageAPI } from '@/hooks/use-page-api'

export default function MarketplacePage() {
  usePageAPI('marketplace')

  return (
    <MainLayout>
      <MarketplaceHomepage />
    </MainLayout>
  )
}
