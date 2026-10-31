'use client'

/**
 * One Marketplace page for everyone (PRD-270 §5.7) — Capabilities › Plugins for a signed-out
 * visitor: the member tab's own PluginCard, header and category pills, over the public read of
 * plugins, in its order (most enabled first).
 *
 * Read-only: nothing is "On", there is no Enabled count, admin menu or Import from GitHub, the card's
 * Enable goes to sign-in carrying the plugin's link, and a click opens the public detail (its
 * version, skills, commands, agents and hooks). The workspace's own plugins are never read.
 */
import { useMemo, useState } from 'react'
import { AnimatePresence } from 'framer-motion'

import { StatusBadge } from '@/components/shared'

import { useMarketplaceAccess } from '../access/marketplace-access'
import { pluginOf } from '../access/public-capability-shapes'
import { PluginCard } from '../marketplace-plugins-tab'
import type { PluginSummary } from '../marketplace-plugins-tab'
import { NO_ITEMS, NO_ITEM_MATCHES } from './copy'
import { PublicItemDetail } from './public-item-detail'
import { ALL, FilterPills, PublicReadState, countedOptions, usePublicEntries } from './public-tab-parts'

const PLUGINS_TAB = 'plugins'
const CATEGORY_LIMIT = 30
const noop = () => undefined

function matches(plugin: PluginSummary, category: string, query: string): boolean {
  if (category !== ALL && plugin.category_name !== category) return false
  const text = [plugin.name, plugin.description, plugin.slug, ...plugin.tags]
  return !query || text.some((part) => part?.toLowerCase().includes(query))
}

export function PublicPluginsTab({ searchQuery }: { searchQuery: string }) {
  const { requireSignIn } = useMarketplaceAccess()
  const { read, entries } = usePublicEntries('plugins')
  const plugins = useMemo(() => entries.map(pluginOf), [entries])
  const categories = useMemo(() => countedOptions(plugins.map((p) => p.category_name), CATEGORY_LIMIT), [plugins])
  const [category, setCategory] = useState(ALL)
  const [openKey, setOpenKey] = useState<string | null>(null)
  const query = searchQuery.trim().toLowerCase()
  const shown = useMemo(() => plugins.filter((plugin) => matches(plugin, category, query)), [plugins, category, query])

  if (read.isLoading || read.isError) return <PublicReadState read={read} />
  return (
    <div className="space-y-6" data-testid="public-plugins-tab">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-xl font-semibold">Capabilities Marketplace</h3>
          <p className="text-sm text-muted-foreground">Teach your agents new skills with community-curated capabilities</p>
        </div>
        <StatusBadge size="sm" status="info">
          {plugins.length} Available
        </StatusBadge>
      </div>
      <FilterPills allLabel="All Capabilities" options={categories} selected={category} onSelect={setCategory} />
      {shown.length === 0 && <p className="text-sm text-muted-foreground">{plugins.length ? NO_ITEM_MATCHES : NO_ITEMS}</p>}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
        <AnimatePresence>
          {shown.map((plugin, index) => (
            <PluginCard
              key={plugin.id}
              plugin={plugin}
              index={index}
              isEnabled={false}
              isEnabling={false}
              onEnable={() => requireSignIn(PLUGINS_TAB, plugin.id)}
              onClick={() => setOpenKey(plugin.id)}
              isAdmin={false}
              isApproving={false}
              isDeleting={false}
              isDeactivating={false}
              onApprove={noop}
              onDeactivate={noop}
              onDelete={noop}
              globalIcon={null}
            />
          ))}
        </AnimatePresence>
      </div>
      {openKey && <PublicItemDetail type="plugins" tab={PLUGINS_TAB} itemKey={openKey} onClose={() => setOpenKey(null)} />}
    </div>
  )
}
