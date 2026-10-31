'use client'

/**
 * One Marketplace page for everyone (PRD-270 §5.7) — the Applications tab for a signed-out visitor:
 * the member tab's own ToolCard, header, category pills and pages of 40, over the public read of
 * applications. Read-only: nothing is "Connected" or "Added" (that is a workspace's), Add to
 * Workspace goes to sign-in carrying the app's link, and Details opens the public detail.
 */
import { useEffect, useMemo, useState } from 'react'
import { AnimatePresence } from 'framer-motion'

import { Badge } from '@/components/ui/badge'
import { EnhancedPagination } from '@/components/ui/pagination'
import type { ComposioApp } from '@/hooks/use-composio-api'

import { useMarketplaceAccess } from '../access/marketplace-access'
import { appOf } from '../access/public-capability-shapes'
import { ToolCard } from '../marketplace-tools-tab'
import { NO_ITEMS, NO_ITEM_MATCHES } from './copy'
import { PublicItemDetail } from './public-item-detail'
import { ALL, FilterPills, PublicReadState, countedOptions, usePublicEntries } from './public-tab-parts'

const TOOLS_TAB = 'tools'
const PAGE_SIZE = 40
const TOP_CATEGORIES = 15

function matches(app: ComposioApp, category: string, query: string): boolean {
  if (category !== ALL && !app.categories.includes(category)) return false
  return (
    !query ||
    app.name.toLowerCase().includes(query) ||
    app.display_name.toLowerCase().includes(query) ||
    Boolean(app.description?.toLowerCase().includes(query))
  )
}

function ToolsHeader({ available }: { available: number }) {
  return (
    <div className="flex items-center justify-between">
      <div>
        <h3 className="text-xl font-semibold">App Integrations</h3>
        <p className="text-sm text-muted-foreground">Connect external apps to give your agents 500+ capabilities</p>
      </div>
      <Badge variant="outline" className="text-[hsl(var(--info))] border-[hsl(var(--info))]/30">
        {available} Available
      </Badge>
    </div>
  )
}

export function PublicToolsTab({ searchQuery }: { searchQuery: string }) {
  const { requireSignIn } = useMarketplaceAccess()
  const { read, entries } = usePublicEntries('applications')
  const apps = useMemo(() => entries.map(appOf), [entries])
  const categories = useMemo(() => countedOptions(apps.flatMap((app) => app.categories), TOP_CATEGORIES), [apps])
  const [category, setCategory] = useState(ALL)
  const [page, setPage] = useState(1)
  const [openKey, setOpenKey] = useState<string | null>(null)
  const query = searchQuery.trim().toLowerCase()
  const shown = useMemo(() => apps.filter((app) => matches(app, category, query)), [apps, category, query])
  useEffect(() => setPage(1), [query, category])

  if (read.isLoading || read.isError) return <PublicReadState read={read} />
  const pages = Math.ceil(shown.length / PAGE_SIZE)
  return (
    <div className="space-y-6" data-testid="public-tools-tab">
      <ToolsHeader available={apps.length} />
      <FilterPills allLabel="All Apps" options={categories} selected={category} onSelect={setCategory} />
      {shown.length === 0 && <p className="text-sm text-muted-foreground">{apps.length ? NO_ITEM_MATCHES : NO_ITEMS}</p>}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
        <AnimatePresence>
          {shown.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE).map((app, index) => (
            <ToolCard
              key={app.name}
              app={app}
              index={index}
              isConnected={false}
              isInWorkspace={false}
              isConnecting={false}
              onConnect={() => requireSignIn(TOOLS_TAB, app.name)}
              onDisconnect={() => undefined}
              onDetails={() => setOpenKey(app.name)}
            />
          ))}
        </AnimatePresence>
      </div>
      {pages > 1 && (
        <EnhancedPagination
          data={{ total: shown.length, skip: (page - 1) * PAGE_SIZE, limit: PAGE_SIZE, pages, current_page: page }}
          onPageChange={setPage}
          className="mt-8"
        />
      )}
      {openKey && <PublicItemDetail type="applications" tab={TOOLS_TAB} itemKey={openKey} onClose={() => setOpenKey(null)} />}
    </div>
  )
}
