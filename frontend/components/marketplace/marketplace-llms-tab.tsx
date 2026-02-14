'use client'

import { useState, useMemo } from 'react'
import {
  Search,
  Zap,
  SortAsc,
  ArrowDownAZ,
  TrendingUp,
  DollarSign,
  Ruler,
} from 'lucide-react'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
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

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const PROVIDER_FILTERS = [
  { id: 'all', name: 'All' },
  { id: 'openai', name: 'OpenAI' },
  { id: 'anthropic', name: 'Anthropic' },
  { id: 'google', name: 'Google' },
  { id: 'openrouter', name: 'OpenRouter' },
]

const CATEGORY_OPTIONS = [
  { value: 'all', label: 'All Categories' },
  { value: 'fast', label: 'Fast' },
  { value: 'balanced', label: 'Balanced' },
  { value: 'premium', label: 'Premium' },
  { value: 'coding', label: 'Coding' },
]

const TIER_OPTIONS = [
  { value: 'all', label: 'All Tiers' },
  { value: 'direct', label: 'Direct' },
  { value: 'aggregator', label: 'Aggregator' },
]

type SortKey = 'cost' | 'context' | 'popularity' | 'name'

const SORT_OPTIONS: { value: SortKey; label: string; icon: typeof DollarSign }[] = [
  { value: 'cost', label: 'Cheapest', icon: DollarSign },
  { value: 'context', label: 'Largest Context', icon: Ruler },
  { value: 'popularity', label: 'Most Popular', icon: TrendingUp },
  { value: 'name', label: 'Name', icon: ArrowDownAZ },
]

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

  // Filter state
  const [selectedProvider, setSelectedProvider] = useState('all')
  const [selectedCategory, setSelectedCategory] = useState('all')
  const [selectedTier, setSelectedTier] = useState('all')
  const [sortBy, setSortBy] = useState<SortKey>('popularity')
  const [localSearch, setLocalSearch] = useState('')

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

  const { data: rawModels = [], isLoading } = useQuery<LLMModel[]>({
    queryKey: [
      'marketplaceLlmModels',
      selectedProvider,
      selectedCategory,
      selectedTier,
      combinedSearch,
    ],
    queryFn: async () => {
      const params = new URLSearchParams()
      if (selectedProvider !== 'all') params.append('provider', selectedProvider)
      if (selectedCategory !== 'all') params.append('category', selectedCategory)
      if (selectedTier !== 'all') params.append('tier', selectedTier)
      if (combinedSearch) params.append('search', combinedSearch)
      const qs = params.toString()
      return apiClient.get(`/api/marketplace/llm/models${qs ? `?${qs}` : ''}`)
    },
  })

  // -----------------------------------------------------------------------
  // Client-side sort
  // -----------------------------------------------------------------------

  const models = useMemo(() => {
    const sorted = [...rawModels]
    switch (sortBy) {
      case 'cost':
        sorted.sort((a, b) => a.input_cost_per_1k - b.input_cost_per_1k)
        break
      case 'context':
        sorted.sort((a, b) => b.context_window - a.context_window)
        break
      case 'popularity':
        sorted.sort((a, b) => b.install_count - a.install_count)
        break
      case 'name':
        sorted.sort((a, b) => a.display_name.localeCompare(b.display_name))
        break
    }
    return sorted
  }, [rawModels, sortBy])

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
    // Refresh the model list after install/uninstall
    queryClient.invalidateQueries({ queryKey: ['marketplaceLlmModels'] })
  }

  // -----------------------------------------------------------------------
  // Render
  // -----------------------------------------------------------------------

  return (
    <div className="space-y-6">
      {/* Filter bar */}
      <div className="space-y-4">
        {/* Provider buttons */}
        <div className="flex flex-wrap gap-2">
          {PROVIDER_FILTERS.map((provider) => (
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
            </Button>
          ))}
        </div>

        {/* Dropdowns + Search + Sort */}
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
          <p className="text-sm mt-1">Try adjusting your filters or search query.</p>
        </div>
      ) : (
        <>
          {/* Result count */}
          <div className="text-xs text-muted-foreground">
            {models.length} model{models.length !== 1 ? 's' : ''} found
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
