'use client'

/**
 * One Marketplace page for everyone (PRD-270 §5.7) — the LLMs tab for a signed-out visitor: the
 * member tab's own LLMModelCard (grid or list), its model count and serving-provider tabs, over the
 * public read of LLM routes, in its order (the member tab's default: most installed first).
 *
 * Read-only: no route is "Added", the card's Add goes to sign-in carrying the route's link
 * (llm-model-card.tsx), and a click opens the public detail (its provider, vendor, context window,
 * price tier and capability ratings). Prices, max output and install counts aren't public, so the
 * card leaves them out; the member tab's sync, compare and catalogue filters need the member reads.
 */
import { useMemo, useState } from 'react'
import { Badge } from '@/components/ui/badge'
import { ViewToggle } from '@/components/shared/view-toggle'
import { useViewMode } from '@/hooks/use-view-mode'

import { llmOf } from '../access/public-capability-shapes'
import { LLMModelCard, routeKeyOf } from '../llm-model-card'
import type { LLMModel } from '../llm-model-card'
import { NO_ITEMS, NO_ITEM_MATCHES } from './copy'
import { PublicItemDetail } from './public-item-detail'
import { ALL, FilterPills, PublicReadState, countedOptions, usePublicEntries } from './public-tab-parts'

const LLMS_TAB = 'llms'
const ALL_PROVIDERS_LIMIT = 50

function matches(model: LLMModel, provider: string, query: string): boolean {
  if (provider !== ALL && model.serving_provider !== provider) return false
  const text = [model.display_name, model.model_id, model.description, model.provider, model.serving_provider_label]
  return !query || text.some((part) => part?.toLowerCase().includes(query))
}

export function PublicLlmsTab({ searchQuery }: { searchQuery: string }) {
  const [viewMode, setViewMode] = useViewMode('mp-llms')
  const { read, entries } = usePublicEntries('llms')
  const models = useMemo(() => entries.map(llmOf), [entries])
  const providers = useMemo(() => countedOptions(models.map((m) => m.serving_provider), ALL_PROVIDERS_LIMIT), [models])
  const labels = useMemo(() => new Map(models.map((m) => [m.serving_provider, m.serving_provider_label])), [models])
  const [provider, setProvider] = useState(ALL)
  const [openKey, setOpenKey] = useState<string | null>(null)
  const query = searchQuery.trim().toLowerCase()
  const shown = useMemo(() => models.filter((model) => matches(model, provider, query)), [models, provider, query])

  if (read.isLoading || read.isError) return <PublicReadState read={read} />
  const providerOptions = providers.map((option) => ({ ...option, name: labels.get(option.id) || option.name }))
  return (
    <div className="space-y-6" data-testid="public-llms-tab">
      <div className="flex items-center justify-between">
        <Badge variant="outline" className="text-xs">
          {models.length} models
        </Badge>
        <ViewToggle value={viewMode} onChange={setViewMode} />
      </div>
      <FilterPills allLabel="All providers" options={providerOptions} selected={provider} onSelect={setProvider} />
      {shown.length === 0 && <p className="text-sm text-muted-foreground">{models.length ? NO_ITEM_MATCHES : NO_ITEMS}</p>}
      <div
        className={
          viewMode === 'list'
            ? 'grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3'
            : 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6'
        }
      >
        {shown.map((model) => (
          <LLMModelCard
            key={routeKeyOf(model)}
            model={model}
            onClick={() => setOpenKey(routeKeyOf(model))}
            viewMode={viewMode === 'list' ? 'list' : 'grid'}
          />
        ))}
      </div>
      {openKey && <PublicItemDetail type="llms" tab={LLMS_TAB} itemKey={openKey} onClose={() => setOpenKey(null)} />}
    </div>
  )
}
