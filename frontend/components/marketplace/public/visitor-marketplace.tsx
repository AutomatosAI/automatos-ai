'use client'

/**
 * PRD-270 US-270-08a/08b (§5.7, K19, K20) — the Marketplace for a signed-out visitor on SaaS.
 *
 * The slim visitor header and the Marketplace's own heading, outside the app shell (no MainLayout:
 * no sidebar, workspace card, notifications or ⌘K workspace search), then the search box and every
 * tab read-only (visitor-tabs.tsx). The tabs read only the public catalogue routes, and the search
 * filters what they loaded, in the browser; nothing here reads or names a workspace.
 */
import { useState } from 'react'

import { PageHeader } from '@/components/shared/page-header'
import { SearchInput } from '@/components/shared/search-input'

import { VISITOR_LEDE, VISITOR_SEARCH } from './copy'
import { VisitorHeader } from './visitor-header'
import { VisitorTabs } from './visitor-tabs'

export function VisitorMarketplace() {
  const [searchQuery, setSearchQuery] = useState('')
  return (
    <div className="min-h-screen bg-background text-foreground">
      <VisitorHeader />
      <main className="mx-auto max-w-6xl space-y-8 px-4 py-8 md:px-6">
        <PageHeader title="Community" titleAccent="Marketplace" eyebrow="Automatos AI · Marketplace" lede={VISITOR_LEDE} />
        <div className="max-w-lg">
          <SearchInput value={searchQuery} onChange={setSearchQuery} placeholder={VISITOR_SEARCH} />
        </div>
        <VisitorTabs searchQuery={searchQuery} />
      </main>
    </div>
  )
}
