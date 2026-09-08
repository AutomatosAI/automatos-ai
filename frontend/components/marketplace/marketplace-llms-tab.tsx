'use client'

import { useState, useMemo } from 'react'
import {
  Search,
  Zap,
  SortAsc,
  ArrowDownAZ,
  DollarSign,
  Ruler,
  RefreshCw,
  Clock,
  Braces,
  Eye,
  Brain,
  Filter,
  X,
} from 'lucide-react'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { ViewToggle } from '@/components/shared/view-toggle'
import { useViewMode } from '@/hooks/use-view-mode'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'
import { toast } from 'sonner'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { ArrowRightLeft, CheckCircle } from 'lucide-react'
import { LLMModelCard } from './llm-model-card'
import { LLMModelDetailModal } from './llm-model-detail-modal'
import type { LLMModel } from './llm-model-card'
import { useProviderRegistry, providerLabel } from '@/hooks/use-provider-registry'
import { useSystemRole } from '@/contexts/role-context'

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const CATEGORY_OPTIONS = [
  { value: 'all', label: 'All Categories' },
  { value: 'fast', label: 'Fast' },
  { value: 'balanced', label: 'Balanced' },
  { value: 'premium', label: 'Premium' },
  { value: 'coding', label: 'Coding' },
  { value: 'reasoning', label: 'Reasoning' },
  { value: 'vision', label: 'Vision' },
  { value: 'free', label: 'Free' },
]

const TIER_OPTIONS = [
  { value: 'all', label: 'All Prices' },
  { value: 'free', label: 'Free' },
  { value: 'budget', label: 'Budget' },
  { value: 'mid', label: 'Mid-range' },
  { value: 'premium', label: 'Premium' },
]

type SortKey = 'cost' | 'context' | 'popularity' | 'name' | 'newest'

const SORT_OPTIONS: { value: SortKey; label: string }[] = [
  { value: 'popularity', label: 'Most Popular' },
  { value: 'cost', label: 'Cheapest First' },
  { value: 'context', label: 'Largest Context' },
  { value: 'newest', label: 'Newest' },
  { value: 'name', label: 'Name A-Z' },
]

// ---------------------------------------------------------------------------
// Catalog response (PRD-236 W1): one row per ROUTE — (serving provider, model)
// ---------------------------------------------------------------------------

interface CatalogResponse {
  models: LLMModel[]
  total: number
  providers: Record<string, number>
  provider_labels: Record<string, string>
  vendors: Record<string, number>
  last_synced: Record<string, string | null>
  syncable: string[]
}

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface MarketplaceLlmsTabProps {
  searchQuery: string
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function MarketplaceLlmsTab({ searchQuery }: MarketplaceLlmsTabProps) {
  const queryClient = useQueryClient()
  const { isAdmin } = useSystemRole()
  const [viewMode, setViewMode] = useViewMode('mp-llms')
  const registry = useProviderRegistry()

  // Filter state — the provider TAB is the serving provider (who you pay), the
  // vendor chip is who made the model (PRD-236 W1)
  const [selectedProvider, setSelectedProvider] = useState('all')
  const [selectedVendor, setSelectedVendor] = useState('all')
  const [selectedCategory, setSelectedCategory] = useState('all')
  const [selectedTier, setSelectedTier] = useState('all')
  const [sortBy, setSortBy] = useState<SortKey>('popularity')
  const [localSearch, setLocalSearch] = useState('')

  // Capability toggles
  const [filterTools, setFilterTools] = useState(false)
  const [filterVision, setFilterVision] = useState(false)
  const [filterReasoning, setFilterReasoning] = useState(false)

  // Detail modal state
  const [selectedModel, setSelectedModel] = useState<LLMModel | null>(null)
  const [detailOpen, setDetailOpen] = useState(false)

  // Compare state
  const [comparing, setComparing] = useState<Set<string>>(new Set())

  // Combined search: parent searchQuery + local
  const combinedSearch = searchQuery || localSearch

  // -----------------------------------------------------------------------
  // Data fetching
  // -----------------------------------------------------------------------

  // Installed ROUTES for this workspace ("provider:model_id" keys)
  const { data: installedData } = useQuery<{ model_ids: string[]; routes?: string[] }>({
    queryKey: ['installed-model-ids'],
    queryFn: () => apiClient.get('/api/marketplace/llm/installed-ids'),
    staleTime: 60 * 1000,
    refetchOnWindowFocus: false,
  })
  const installedRoutes = useMemo(
    () => new Set(installedData?.routes ?? []),
    [installedData]
  )

  const { data: response, isLoading } = useQuery<CatalogResponse>({
    queryKey: [
      'marketplaceLlmCatalog',
      selectedProvider,
      selectedVendor,
      selectedCategory,
      selectedTier,
      combinedSearch,
      sortBy,
      filterTools,
      filterVision,
      filterReasoning,
    ],
    queryFn: async () => {
      const params = new URLSearchParams()
      if (selectedProvider !== 'all') params.append('provider', selectedProvider)
      if (selectedVendor !== 'all') params.append('vendor', selectedVendor)
      if (selectedCategory !== 'all') params.append('category', selectedCategory)
      if (selectedTier !== 'all') params.append('price_tier', selectedTier)
      if (combinedSearch) params.append('search', combinedSearch)
      if (sortBy) params.append('sort_by', sortBy)
      if (filterTools) params.append('supports_tools', 'true')
      if (filterVision) params.append('supports_vision', 'true')
      if (filterReasoning) params.append('category', 'reasoning')
      params.append('limit', '500')
      const qs = params.toString()
      return apiClient.get(`/api/marketplace/llm/catalog${qs ? `?${qs}` : ''}`)
    },
    staleTime: 5 * 60 * 1000,
    refetchOnWindowFocus: false,
  })

  // Sync one provider's catalogue (admin). OpenRouter and NVIDIA are syncable.
  const syncMutation = useMutation({
    mutationFn: (slug: string) => apiClient.post(`/api/marketplace/llm/sync/${encodeURIComponent(slug)}`),
    onSuccess: (_data, slug) => {
      toast.success(`${providerLabel(registry, slug)} catalogue synced`)
      queryClient.invalidateQueries({ queryKey: ['marketplaceLlmCatalog'] })
      queryClient.invalidateQueries({ queryKey: ['openrouterModels'] })
    },
    onError: (err: Error, slug) => {
      toast.error(`${providerLabel(registry, slug)} sync failed`, { description: err.message })
    },
  })

  const rawModels = response?.models ?? []
  const providerCounts = response?.providers ?? {}
  const providerLabels = response?.provider_labels ?? {}
  const vendorCounts = response?.vendors ?? {}
  const syncable = response?.syncable ?? ['openrouter', 'nvidia']
  const lastSyncedByProvider = response?.last_synced ?? {}
  const totalCount = response?.total ?? 0
  const catalogTotal = useMemo(
    () => Object.values(providerCounts).reduce((a, b) => a + b, 0),
    [providerCounts]
  )

  // Provider TABS: every registered chat provider, in registry order, with the
  // number of routes it serves. Providers with no rows yet still get a tab so
  // the "Sync" action has somewhere to live.
  const providerTabs = useMemo(() => {
    const ordered = registry.providers.filter((p) => p.chat).map((p) => p.slug)
    const extra = Object.keys(providerCounts).filter((slug) => !ordered.includes(slug))
    const tabs = [...ordered, ...extra]
      .filter((slug) => (providerCounts[slug] ?? 0) > 0 || syncable.includes(slug))
      .map((slug) => ({
        id: slug,
        name: providerLabels[slug] || providerLabel(registry, slug),
        count: providerCounts[slug] ?? 0,
      }))
    return [{ id: 'all', name: 'All providers', count: catalogTotal }, ...tabs]
  }, [registry, providerCounts, providerLabels, syncable, catalogTotal])

  // Vendor chips within the selected provider tab
  const vendorFilters = useMemo(() => {
    const vendors = Object.entries(vendorCounts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 16)
      .map(([id, count]) => ({ id, name: id.charAt(0).toUpperCase() + id.slice(1), count }))
    return [{ id: 'all', name: 'All vendors', count: totalCount }, ...vendors]
  }, [vendorCounts, totalCount])

  // The catalog rows already match the card shape; installed is per ROUTE
  const models = useMemo(
    () =>
      rawModels.map((m) => ({
        ...m,
        is_installed: installedRoutes.has(`${m.serving_provider || m.provider}:${m.model_id}`),
      })),
    [rawModels, installedRoutes]
  )

  // Which providers may be synced from the current tab
  const syncTargets = useMemo(
    () => (selectedProvider === 'all' ? syncable : syncable.filter((slug) => slug === selectedProvider)),
    [selectedProvider, syncable]
  )
  const lastSynced = selectedProvider === 'all'
    ? (lastSyncedByProvider['openrouter'] ?? null)
    : (lastSyncedByProvider[selectedProvider] ?? null)

  // Active filter count (for indicator)
  const activeFilterCount = [
    selectedVendor !== 'all',
    selectedCategory !== 'all',
    selectedTier !== 'all',
    filterTools,
    filterVision,
    filterReasoning,
  ].filter(Boolean).length

  // -----------------------------------------------------------------------
  // Handlers
  // -----------------------------------------------------------------------

  const handleModelClick = (model: LLMModel) => {
    setSelectedModel(model)
    setDetailOpen(true)
  }

  const handleCompareToggle = (modelId: string) => {
    setComparing((prev) => {
      const next = new Set(prev)
      if (next.has(modelId)) {
        next.delete(modelId)
      } else {
        if (next.size >= 4) {
          toast.warning('Compare up to 4 models at a time')
          return prev
        }
        next.add(modelId)
      }
      return next
    })
  }

  const handleInstall = () => {
    queryClient.invalidateQueries({ queryKey: ['marketplaceLlmCatalog'] })
    queryClient.invalidateQueries({ queryKey: ['marketplaceLlmModels'] })
    queryClient.invalidateQueries({ queryKey: ['installed-model-ids'] })
    queryClient.invalidateQueries({ queryKey: ['workspace-models'] })
  }

  const clearAllFilters = () => {
    setSelectedVendor('all')
    setSelectedCategory('all')
    setSelectedTier('all')
    setFilterTools(false)
    setFilterVision(false)
    setFilterReasoning(false)
    setLocalSearch('')
  }

  // Format last synced time
  const lastSyncedText = useMemo(() => {
    if (!lastSynced) return null
    const d = new Date(lastSynced)
    const now = new Date()
    const diffMs = now.getTime() - d.getTime()
    const diffMins = Math.floor(diffMs / 60000)
    if (diffMins < 1) return 'just now'
    if (diffMins < 60) return `${diffMins}m ago`
    const diffHours = Math.floor(diffMins / 60)
    if (diffHours < 24) return `${diffHours}h ago`
    const diffDays = Math.floor(diffHours / 24)
    return `${diffDays}d ago`
  }, [lastSynced])

  // -----------------------------------------------------------------------
  // Render
  // -----------------------------------------------------------------------

  return (
    <div className="space-y-6">
      {/* Header with stats + sync */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Badge variant="outline" className="text-xs">
            {totalCount} models
          </Badge>
          {lastSyncedText && (
            <span className="text-[11px] text-muted-foreground flex items-center gap-1">
              <Clock className="w-3 h-3" />
              Synced {lastSyncedText}
            </span>
          )}
        </div>
        <div className="flex items-center gap-3">
          <ViewToggle value={viewMode} onChange={setViewMode} />
          {isAdmin && syncTargets.map((slug) => (
            <Button
              key={slug}
              variant="outline"
              size="sm"
              onClick={() => syncMutation.mutate(slug)}
              disabled={syncMutation.isLoading}
              className="gap-2 text-muted-foreground hover:text-foreground"
              data-testid={`sync-${slug}`}
            >
              <RefreshCw className={`w-3.5 h-3.5 ${syncMutation.isLoading && syncMutation.variables === slug ? 'animate-spin' : ''}`} />
              {syncMutation.isLoading && syncMutation.variables === slug ? 'Syncing...' : `Sync ${providerLabels[slug] || providerLabel(registry, slug)}`}
            </Button>
          ))}
        </div>
      </div>

      {/* Filter bar */}
      <div className="space-y-4">
        {/* Provider TABS — who serves (and bills) the call. "Kimi K3 · NVIDIA" and
            "Kimi K3 · OpenRouter" are different routes with different prices. */}
        <div
          role="tablist"
          aria-label="Serving provider"
          data-testid="provider-tabs"
          className="flex gap-1 overflow-x-auto pb-1 border-b border-border/40 scrollbar-thin scrollbar-thumb-secondary scrollbar-track-transparent"
        >
          {providerTabs.map((tab) => (
            <button
              key={tab.id}
              role="tab"
              aria-selected={selectedProvider === tab.id}
              onClick={() => {
                setSelectedProvider(tab.id)
                setSelectedVendor('all')
              }}
              className={`whitespace-nowrap flex-shrink-0 px-3 py-2 text-sm border-b-2 -mb-px transition-colors ${
                selectedProvider === tab.id
                  ? 'border-primary text-foreground font-semibold'
                  : 'border-transparent text-muted-foreground hover:text-foreground'
              }`}
            >
              {tab.name}
              <span className="ml-1.5 text-[10px] opacity-60">{tab.count}</span>
            </button>
          ))}
        </div>

        {/* Vendor chips within the tab — who made the model */}
        <div className="flex gap-2 overflow-x-auto pb-1 scrollbar-thin scrollbar-thumb-secondary scrollbar-track-transparent">
          {vendorFilters.map((vendor) => (
            <Button
              key={vendor.id}
              variant={selectedVendor === vendor.id ? 'default' : 'outline'}
              size="sm"
              onClick={() => setSelectedVendor(vendor.id)}
              className={`whitespace-nowrap flex-shrink-0 ${
                selectedVendor === vendor.id
                  ? 'bg-secondary border-primary/50 text-foreground font-semibold'
                  : 'border-secondary text-muted-foreground hover:bg-secondary'
              }`}
            >
              {vendor.name}
              <span className="ml-1.5 text-[10px] opacity-60">{vendor.count}</span>
            </Button>
          ))}
        </div>

        {/* Dropdowns + Capability toggles + Search + Sort */}
        <div className="flex flex-wrap items-center gap-3">
          {/* Category dropdown */}
          <Select value={selectedCategory} onValueChange={setSelectedCategory}>
            <SelectTrigger className="w-[160px] h-9 text-xs">
              <SelectValue placeholder="Category" />
            </SelectTrigger>
            <SelectContent>
              {CATEGORY_OPTIONS.map((opt) => (
                <SelectItem key={opt.value} value={opt.value}>
                  {opt.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          {/* Tier dropdown */}
          <Select value={selectedTier} onValueChange={setSelectedTier}>
            <SelectTrigger className="w-[140px] h-9 text-xs">
              <SelectValue placeholder="Tier" />
            </SelectTrigger>
            <SelectContent>
              {TIER_OPTIONS.map((opt) => (
                <SelectItem key={opt.value} value={opt.value}>
                  {opt.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          {/* Capability toggle buttons */}
          <Button
            variant={filterTools ? 'default' : 'outline'}
            size="sm"
            onClick={() => setFilterTools(!filterTools)}
            className={`h-9 text-xs gap-1.5 ${
              filterTools
                ? 'bg-[hsl(var(--info))]/15 border-[hsl(var(--info))]/40 text-[hsl(var(--info))]'
                : 'border-secondary text-muted-foreground'
            }`}
          >
            <Braces className="w-3 h-3" />
            Tools
          </Button>

          <Button
            variant={filterVision ? 'default' : 'outline'}
            size="sm"
            onClick={() => setFilterVision(!filterVision)}
            className={`h-9 text-xs gap-1.5 ${
              filterVision
                ? 'bg-[hsl(var(--agent))]/15 border-[hsl(var(--agent))]/40 text-[hsl(var(--agent))]'
                : 'border-secondary text-muted-foreground'
            }`}
          >
            <Eye className="w-3 h-3" />
            Vision
          </Button>

          <Button
            variant={filterReasoning ? 'default' : 'outline'}
            size="sm"
            onClick={() => setFilterReasoning(!filterReasoning)}
            className={`h-9 text-xs gap-1.5 ${
              filterReasoning
                ? 'bg-agent/15 border-agent/40 text-agent'
                : 'border-secondary text-muted-foreground'
            }`}
          >
            <Brain className="w-3 h-3" />
            Reasoning
          </Button>

          {/* Sort */}
          <Select value={sortBy} onValueChange={(v) => setSortBy(v as SortKey)}>
            <SelectTrigger className="w-[160px] h-9 text-xs">
              <SortAsc className="w-3 h-3 mr-1.5" />
              <SelectValue placeholder="Sort by" />
            </SelectTrigger>
            <SelectContent>
              {SORT_OPTIONS.map((opt) => (
                <SelectItem key={opt.value} value={opt.value}>
                  {opt.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          {/* Search */}
          <div className="relative flex-1 min-w-[200px]">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-muted-foreground" />
            <Input
              placeholder="Search models..."
              value={localSearch}
              onChange={(e) => setLocalSearch(e.target.value)}
              className="pl-9 h-9 text-xs"
            />
          </div>
        </div>

        {/* Active filters indicator + clear */}
        {activeFilterCount > 0 && (
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="text-[10px] gap-1">
              <Filter className="w-2.5 h-2.5" />
              {activeFilterCount} filter{activeFilterCount > 1 ? 's' : ''} active
            </Badge>
            <Button
              variant="ghost"
              size="sm"
              onClick={clearAllFilters}
              className="h-6 text-[10px] text-muted-foreground hover:text-foreground px-2"
            >
              <X className="w-2.5 h-2.5 mr-1" />
              Clear all
            </Button>
          </div>
        )}

        {/* Model Comparison Panel */}
        {comparing.size > 0 && (() => {
          const COMPARISON_COLORS = ['#3b82f6', '#f59e0b', '#10b981', '#ef4444']
          const comparedModels = models.filter(m => comparing.has(m.model_id))
          return (
            <Card className="glass-card">
              <CardHeader className="pb-3">
                <CardTitle className="flex items-center justify-between">
                  <span className="flex items-center gap-2 text-sm">
                    <ArrowRightLeft className="w-4 h-4 text-[hsl(var(--info))]" />
                    Model Comparison
                  </span>
                  <Button size="sm" variant="ghost" className="h-7 text-xs text-muted-foreground"
                    onClick={() => setComparing(new Set())}>
                    <X className="w-3 h-3 mr-1" /> Clear
                  </Button>
                </CardTitle>
                <div className="flex flex-wrap gap-2">
                  {comparedModels.map((m, idx) => (
                    <Badge key={m.model_id} variant="secondary"
                      className="font-mono text-xs flex items-center gap-1.5 py-1"
                      style={{ borderColor: COMPARISON_COLORS[idx] }}>
                      <span className="w-2 h-2 rounded-full" style={{ background: COMPARISON_COLORS[idx] }} />
                      {m.display_name}
                      <button onClick={() => handleCompareToggle(m.model_id)}
                        className="ml-0.5 hover:text-destructive transition-colors">
                        <X className="w-3 h-3" />
                      </button>
                    </Badge>
                  ))}
                  {comparing.size < 4 && (
                    <span className="text-xs text-muted-foreground self-center">
                      Click &quot;Compare&quot; on cards to add (up to 4)
                    </span>
                  )}
                </div>
              </CardHeader>
              {comparedModels.length >= 2 && (
                <CardContent>
                  <div className="overflow-x-auto">
                    <table className="w-full">
                      <thead>
                        <tr className="border-b border-border/50">
                          <th className="text-left p-3 text-[11px] font-medium text-muted-foreground uppercase tracking-wider">Metric</th>
                          {comparedModels.map((m, idx) => (
                            <th key={m.model_id} className="text-right p-3 text-[11px] font-medium uppercase tracking-wider"
                              style={{ color: COMPARISON_COLORS[idx] }}>
                              {m.display_name}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {[
                          { label: 'Route', render: (m: LLMModel) => m.route_label || `${m.display_name} · ${m.serving_provider || m.provider}` },
                          { label: 'Vendor', render: (m: LLMModel) => m.provider },
                          { label: 'Context Window', render: (m: LLMModel) => {
                            const n = m.context_window
                            return n >= 1_000_000 ? `${(n / 1_000_000).toFixed(1)}M` : n >= 1_000 ? `${(n / 1_000).toFixed(0)}K` : String(n)
                          }},
                          { label: 'Max Output', render: (m: LLMModel) => {
                            const n = m.max_output_tokens
                            return n >= 1_000_000 ? `${(n / 1_000_000).toFixed(1)}M` : n >= 1_000 ? `${(n / 1_000).toFixed(0)}K` : String(n)
                          }},
                          { label: 'Input Cost/1M', render: (m: LLMModel) => {
                            const per1M = m.input_cost_per_1k * 1000
                            return per1M === 0 ? 'Free' : per1M < 1 ? `$${per1M.toFixed(3)}` : `$${per1M.toFixed(2)}`
                          }},
                          { label: 'Output Cost/1M', render: (m: LLMModel) => {
                            const per1M = m.output_cost_per_1k * 1000
                            return per1M === 0 ? 'Free' : per1M < 1 ? `$${per1M.toFixed(3)}` : `$${per1M.toFixed(2)}`
                          }},
                          { label: 'Functions', render: (m: LLMModel) => m.supports_functions ? 'Yes' : 'No' },
                          { label: 'Vision', render: (m: LLMModel) => m.supports_vision ? 'Yes' : 'No' },
                          { label: 'Streaming', render: (m: LLMModel) => m.supports_streaming ? 'Yes' : 'No' },
                          { label: 'Status', render: (m: LLMModel) => (
                            <span className="inline-flex items-center gap-1">
                              {m.is_installed && <CheckCircle className="w-3 h-3 text-[hsl(var(--success))]" />}
                              {m.is_installed ? 'Added' : 'Not added'}
                            </span>
                          )},
                        ].map((row) => (
                          <tr key={row.label} className="border-b border-border/20">
                            <td className="p-3 text-xs text-muted-foreground">{row.label}</td>
                            {comparedModels.map(m => (
                              <td key={m.model_id} className="p-3 text-sm text-right tabular-nums">
                                {row.render(m)}
                              </td>
                            ))}
                          </tr>
                        ))}
                        <tr>
                          <td className="p-3 text-xs text-muted-foreground">Tags</td>
                          {comparedModels.map(m => (
                            <td key={m.model_id} className="p-3 text-right">
                              <div className="flex flex-wrap gap-1 justify-end">
                                {(m.recommended_for || []).slice(0, 3).map(tag => (
                                  <Badge key={tag} variant="outline" className="text-[10px] px-1.5 py-0">{tag}</Badge>
                                ))}
                              </div>
                            </td>
                          ))}
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </CardContent>
              )}
            </Card>
          )
        })()}
      </div>

      {/* Results */}
      {isLoading ? (
        viewMode === 'list' ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {[...Array(6)].map((_, i) => (
              <div key={i} className="h-16 glass-card animate-pulse rounded-xl" />
            ))}
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            {[...Array(8)].map((_, i) => (
              <div key={i} className="h-72 glass-card animate-pulse rounded-xl" />
            ))}
          </div>
        )
      ) : models.length === 0 ? (
        <div className="text-center py-16 text-muted-foreground">
          <Zap className="w-12 h-12 mx-auto mb-4 opacity-50" />
          <p className="font-medium">No LLM models found</p>
          <p className="text-sm mt-1">
            {totalCount === 0
              ? `No routes in this catalogue yet. ${isAdmin ? 'Sync a provider to load its models.' : 'Ask an admin to sync a provider.'}`
              : 'Try adjusting your filters or search query.'}
          </p>
          {totalCount === 0 && isAdmin && syncTargets.map((slug) => (
            <Button
              key={slug}
              variant="outline"
              className="mt-4 mr-2"
              onClick={() => syncMutation.mutate(slug)}
              disabled={syncMutation.isLoading}
            >
              <RefreshCw className={`w-4 h-4 mr-2 ${syncMutation.isLoading && syncMutation.variables === slug ? 'animate-spin' : ''}`} />
              {syncMutation.isLoading && syncMutation.variables === slug ? 'Syncing...' : `Sync ${providerLabels[slug] || providerLabel(registry, slug)} models`}
            </Button>
          ))}
        </div>
      ) : (
        <>
          {/* Result count */}
          <div className="text-xs text-muted-foreground">
            Showing {models.length} of {totalCount} model{totalCount !== 1 ? 's' : ''}
          </div>

          {viewMode === 'list' ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
              {models.map((model) => (
                <LLMModelCard
                  key={model.id}
                  model={model}
                  onInstall={handleInstall}
                  onCompareToggle={handleCompareToggle}
                  isComparing={comparing.has(model.model_id)}
                  onClick={() => handleModelClick(model)}
                  viewMode="list"
                />
              ))}
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
              {models.map((model) => (
                <LLMModelCard
                  key={model.id}
                  model={model}
                  onInstall={handleInstall}
                  onCompareToggle={handleCompareToggle}
                  isComparing={comparing.has(model.model_id)}
                  onClick={() => handleModelClick(model)}
                />
              ))}
            </div>
          )}
        </>
      )}

      {/* Detail modal */}
      <LLMModelDetailModal
        model={selectedModel}
        open={detailOpen}
        onOpenChange={setDetailOpen}
        onInstall={handleInstall}
      />
    </div>
  )
}
