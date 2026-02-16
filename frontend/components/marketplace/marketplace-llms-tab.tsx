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
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'
import { toast } from 'sonner'
import { LLMModelCard } from './llm-model-card'
import { LLMModelDetailModal } from './llm-model-detail-modal'
import type { LLMModel } from './llm-model-card'
import { useSyncOpenRouterCache } from '@/hooks/use-openrouter-api'
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
  { value: 'all', label: 'All Tiers' },
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
// Types for OpenRouter cache response
// ---------------------------------------------------------------------------

interface OpenRouterModel {
  id: number
  model_id: string
  slug: string | null
  display_name: string
  description: string | null
  provider: string
  prompt_cost: number
  completion_cost: number
  image_cost: number
  request_cost: number
  cache_read_cost: number
  cache_write_cost: number
  context_length: number
  max_completion_tokens: number
  modality: string | null
  input_modalities: string[]
  output_modalities: string[]
  tokenizer: string | null
  supports_tools: boolean
  supports_vision: boolean
  supports_streaming: boolean
  supports_json_mode: boolean
  supports_reasoning: boolean
  category: string | null
  tier: string | null
  tags: string[]
  status: string
  is_moderated: boolean
  last_synced_at: string | null
}

interface OpenRouterMarketplaceResponse {
  models: OpenRouterModel[]
  total: number
  providers: Record<string, number>
  last_synced: string | null
}

// Adapt OpenRouter model to the existing LLMModel card interface
function adaptToLLMModel(m: OpenRouterModel, installedIds?: Set<string>): LLMModel {
  return {
    id: m.id,
    provider: m.provider,
    model_id: m.model_id,
    display_name: m.display_name,
    model_family: null,
    description: m.description,
    context_window: m.context_length,
    max_output_tokens: m.max_completion_tokens,
    // Convert per-token cost to per-1K-token cost
    input_cost_per_1k: m.prompt_cost * 1000,
    output_cost_per_1k: m.completion_cost * 1000,
    capabilities: {},
    recommended_for: m.tags || [],
    supports_functions: m.supports_tools,
    supports_vision: m.supports_vision,
    supports_streaming: m.supports_streaming,
    status: m.status,
    tier: m.tier,
    category: m.category,
    tags: m.tags || [],
    is_featured: false,
    is_default: false,
    requires_plan: null,
    install_count: 0,
    is_installed: installedIds ? installedIds.has(m.model_id) : false,
  }
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
  const syncMutation = useSyncOpenRouterCache()

  // Filter state
  const [selectedProvider, setSelectedProvider] = useState('all')
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

  // Fetch installed model IDs for this workspace (lightweight)
  const { data: installedData } = useQuery<{ model_ids: string[] }>({
    queryKey: ['installed-model-ids'],
    queryFn: () => apiClient.get('/api/marketplace/llm/installed-ids'),
    staleTime: 60 * 1000,
    refetchOnWindowFocus: false,
  })
  const installedIds = useMemo(
    () => new Set(installedData?.model_ids ?? []),
    [installedData]
  )

  const { data: response, isLoading } = useQuery<OpenRouterMarketplaceResponse>({
    queryKey: [
      'openrouterModels',
      selectedProvider,
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
      if (selectedCategory !== 'all') params.append('category', selectedCategory)
      if (selectedTier !== 'all') params.append('tier', selectedTier)
      if (combinedSearch) params.append('search', combinedSearch)
      if (sortBy) params.append('sort_by', sortBy)
      if (filterTools) params.append('supports_tools', 'true')
      if (filterVision) params.append('supports_vision', 'true')
      if (filterReasoning) params.append('supports_reasoning', 'true')
      params.append('limit', '500')
      const qs = params.toString()

      // Try OpenRouter cache first, fall back to legacy LLM endpoint
      try {
        return await apiClient.get(`/api/openrouter/models${qs ? `?${qs}` : ''}`)
      } catch {
        // Fallback: legacy endpoint (returns LLMModel[] array, not wrapped)
        const legacyParams = new URLSearchParams()
        if (selectedProvider !== 'all') legacyParams.append('provider', selectedProvider)
        if (selectedCategory !== 'all') legacyParams.append('category', selectedCategory)
        if (selectedTier !== 'all') legacyParams.append('tier', selectedTier)
        if (combinedSearch) legacyParams.append('search', combinedSearch)
        const lqs = legacyParams.toString()
        const legacyModels = await apiClient.get(`/api/marketplace/llm/models${lqs ? `?${lqs}` : ''}`)

        // Wrap in the expected response shape
        const providerMap: Record<string, number> = {}
        for (const m of legacyModels) {
          providerMap[m.provider] = (providerMap[m.provider] || 0) + 1
        }
        return {
          models: legacyModels.map((m: any) => ({
            id: m.id,
            model_id: m.model_id,
            slug: null,
            display_name: m.display_name,
            description: m.description,
            provider: m.provider,
            prompt_cost: (m.input_cost_per_1k || 0) / 1000,
            completion_cost: (m.output_cost_per_1k || 0) / 1000,
            image_cost: 0,
            request_cost: 0,
            cache_read_cost: 0,
            cache_write_cost: 0,
            context_length: m.context_window || 0,
            max_completion_tokens: m.max_output_tokens || 0,
            modality: null,
            input_modalities: [],
            output_modalities: [],
            tokenizer: null,
            supports_tools: m.supports_functions || false,
            supports_vision: m.supports_vision || false,
            supports_streaming: m.supports_streaming ?? true,
            supports_json_mode: false,
            supports_reasoning: false,
            category: m.category,
            tier: m.tier,
            tags: m.tags || [],
            status: m.status || 'active',
            is_moderated: false,
            last_synced_at: null,
          })),
          total: legacyModels.length,
          providers: providerMap,
          last_synced: null,
        } as OpenRouterMarketplaceResponse
      }
    },
    staleTime: 5 * 60 * 1000,
    refetchOnWindowFocus: false,
  })

  const rawModels = response?.models ?? []
  const providerCounts = response?.providers ?? {}
  const lastSynced = response?.last_synced ?? null
  const totalCount = response?.total ?? 0

  // Build dynamic provider filter chips from actual data
  const providerFilters = useMemo(() => {
    const providers = Object.entries(providerCounts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 12)
      .map(([id, count]) => ({ id, name: id.charAt(0).toUpperCase() + id.slice(1), count }))
    return [{ id: 'all', name: 'All', count: totalCount }, ...providers]
  }, [providerCounts, totalCount])

  // Adapt to existing card format (with installed status)
  const models = useMemo(
    () => rawModels.map((m) => adaptToLLMModel(m, installedIds)),
    [rawModels, installedIds]
  )

  // Active filter count (for indicator)
  const activeFilterCount = [
    selectedProvider !== 'all',
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
    queryClient.invalidateQueries({ queryKey: ['openrouterModels'] })
    queryClient.invalidateQueries({ queryKey: ['marketplaceLlmModels'] })
    queryClient.invalidateQueries({ queryKey: ['installed-model-ids'] })
    queryClient.invalidateQueries({ queryKey: ['workspace-models'] })
  }

  const clearAllFilters = () => {
    setSelectedProvider('all')
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
        {isAdmin && (
          <Button
            variant="outline"
            size="sm"
            onClick={() => syncMutation.mutate()}
            disabled={syncMutation.isPending}
            className="gap-2 text-muted-foreground hover:text-foreground"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${syncMutation.isPending ? 'animate-spin' : ''}`} />
            {syncMutation.isPending ? 'Syncing...' : 'Sync OpenRouter'}
          </Button>
        )}
      </div>

      {/* Filter bar */}
      <div className="space-y-4">
        {/* Provider buttons (dynamic from cache) */}
        <div className="flex flex-wrap gap-2">
          {providerFilters.map((provider) => (
            <Button
              key={provider.id}
              variant={selectedProvider === provider.id ? 'default' : 'outline'}
              size="sm"
              onClick={() => setSelectedProvider(provider.id)}
              className={
                selectedProvider === provider.id
                  ? 'bg-secondary border-primary/50 text-foreground font-semibold'
                  : 'border-secondary text-muted-foreground hover:bg-secondary'
              }
            >
              {provider.name}
              <span className="ml-1.5 text-[10px] opacity-60">{provider.count}</span>
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
                ? 'bg-purple-500/15 border-purple-500/40 text-purple-400'
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

        {/* Compare bar */}
        {comparing.size > 0 && (
          <div className="flex items-center gap-3 p-3 rounded-lg border border-[hsl(var(--info))]/30 bg-[hsl(var(--info))]/5">
            <span className="text-xs text-[hsl(var(--info))] font-medium">
              Comparing {comparing.size} model{comparing.size > 1 ? 's' : ''}
            </span>
            <Button
              size="sm"
              variant="outline"
              className="h-7 text-xs border-[hsl(var(--info))]/30 text-[hsl(var(--info))]"
              onClick={() => setComparing(new Set())}
            >
              Clear
            </Button>
          </div>
        )}
      </div>

      {/* Results */}
      {isLoading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          {[...Array(8)].map((_, i) => (
            <div key={i} className="h-72 glass-card animate-pulse rounded-xl" />
          ))}
        </div>
      ) : models.length === 0 ? (
        <div className="text-center py-16 text-muted-foreground">
          <Zap className="w-12 h-12 mx-auto mb-4 opacity-50" />
          <p className="font-medium">No LLM models found</p>
          <p className="text-sm mt-1">
            {totalCount === 0
              ? 'Cache is empty. Click "Sync OpenRouter" to load models.'
              : 'Try adjusting your filters or search query.'}
          </p>
          {totalCount === 0 && isAdmin && (
            <Button
              variant="outline"
              className="mt-4"
              onClick={() => syncMutation.mutate()}
              disabled={syncMutation.isPending}
            >
              <RefreshCw className={`w-4 h-4 mr-2 ${syncMutation.isPending ? 'animate-spin' : ''}`} />
              {syncMutation.isPending ? 'Syncing...' : 'Sync OpenRouter Models'}
            </Button>
          )}
        </div>
      ) : (
        <>
          {/* Result count */}
          <div className="text-xs text-muted-foreground">
            Showing {models.length} of {totalCount} model{totalCount !== 1 ? 's' : ''}
          </div>

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
