'use client'

import { useState } from 'react'
import {
  Star,
  Download,
  Loader2,
  Eye,
  Braces,
  Radio,
  CheckCircle,
  Shield,
  ArrowRightLeft,
} from 'lucide-react'
import { Card, CardContent, CardHeader } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { apiClient } from '@/lib/api-client'
import { useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface LLMModel {
  id: number
  /** The VENDOR of the model ('openai', 'moonshotai', 'deepseek-ai'…). */
  provider: string
  vendor?: string
  /** PRD-236 W1: the ROUTE that serves this row ('openrouter', 'nvidia', 'openai'…). */
  serving_provider?: string
  serving_provider_label?: string
  route_label?: string
  is_free?: boolean
  price_tier?: string
  key_available?: boolean
  sourcing?: string | null
  terms_note?: string | null
  rate_limit_note?: string | null
  model_id: string
  display_name: string
  model_family: string | null
  description: string | null
  context_window: number
  max_output_tokens: number
  input_cost_per_1k: number
  output_cost_per_1k: number
  capabilities: Record<string, string>
  recommended_for: string[]
  supports_functions: boolean
  supports_vision: boolean
  supports_streaming: boolean
  status: string
  /** Legacy name of `sourcing`; W1 responses no longer send it. */
  tier?: string | null
  category: string | null
  tags: string[]
  is_featured: boolean
  is_default: boolean
  requires_plan: string | null
  install_count: number
  is_installed: boolean
}

export interface LLMModelCardProps {
  model: LLMModel
  onInstall?: (modelId: string) => void
  onCompareToggle?: (modelId: string) => void
  isComparing?: boolean
  onClick?: () => void
  viewMode?: 'grid' | 'list'
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const PROVIDER_COLORS: Record<string, string> = {
  openai: 'bg-success/15 text-success border-success/30',
  anthropic: 'bg-warning/15 text-warning border-warning/30',
  google: 'bg-info/15 text-info border-info/30',
  openrouter: 'bg-agent/15 text-agent border-agent/30',
  nvidia: 'bg-lime-500/15 text-lime-400 border-lime-500/30',
  deepseek: 'bg-indigo-500/15 text-indigo-400 border-indigo-500/30',
  moonshotai: 'bg-violet-500/15 text-violet-400 border-violet-500/30',
  meta: 'bg-sky-500/15 text-sky-400 border-sky-500/30',
  mistral: 'bg-warning/15 text-warning border-warning/30',
  cohere: 'bg-pink-500/15 text-pink-400 border-pink-500/30',
}

const CAPABILITY_RATING: Record<string, { label: string; width: string; color: string }> = {
  excellent: { label: 'Excellent', width: 'w-full', color: 'bg-[hsl(var(--success))]' },
  good: { label: 'Good', width: 'w-3/4', color: 'bg-[hsl(var(--info))]' },
  moderate: { label: 'Moderate', width: 'w-1/2', color: 'bg-[hsl(var(--primary))]' },
  basic: { label: 'Basic', width: 'w-1/4', color: 'bg-muted-foreground' },
  none: { label: 'None', width: 'w-0', color: 'bg-muted' },
}

function formatTokenCount(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(n % 1_000_000 === 0 ? 0 : 1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(n % 1_000 === 0 ? 0 : 1)}K`
  return n.toString()
}

function formatCostPer1M(costPer1K: number): string {
  const per1M = costPer1K * 1000
  if (per1M === 0) return 'Free'
  if (per1M < 0.01) return `$${per1M.toFixed(4)}`
  if (per1M < 1) return `$${per1M.toFixed(3)}`
  return `$${per1M.toFixed(2)}`
}

function formatInstallCount(count: number): string {
  if (count >= 1_000_000) return `${(count / 1_000_000).toFixed(1)}M`
  if (count >= 1_000) return `${(count / 1_000).toFixed(1)}k`
  return count.toString()
}

function providerBadgeClass(provider: string): string {
  const key = provider.toLowerCase()
  return PROVIDER_COLORS[key] || 'bg-secondary text-muted-foreground border-border'
}

/** PRD-236 W1: the route badge — who serves the call — plus the vendor when it differs. */
export function routeBadges(model: LLMModel): { route: string; routeKey: string; vendor: string | null } {
  const routeKey = model.serving_provider || model.provider
  const route = model.serving_provider_label || routeKey
  const vendor = model.serving_provider && model.provider !== model.serving_provider ? model.provider : null
  return { route, routeKey, vendor }
}

const SOURCING_LABELS: Record<string, string> = {
  direct: 'Direct API',
  aggregator: 'Aggregator',
  hosted_open: 'Hosted open model',
  byok_only: 'BYO key',
}

export function sourcingLabel(model: LLMModel): string | null {
  const key = model.sourcing ?? model.tier ?? null
  if (!key) return null
  return SOURCING_LABELS[key] || key
}

/** The install/uninstall path for THIS route (the provider query pins the row). */
export function routeActionPath(model: LLMModel, action: 'install' | 'uninstall'): string {
  const encodedId = encodeURIComponent(model.model_id)
  const provider = model.serving_provider ? `?provider=${encodeURIComponent(model.serving_provider)}` : ''
  return `/api/marketplace/llm/models/${encodedId}/${action}${provider}`
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function LLMModelCard({
  model,
  onInstall,
  onCompareToggle,
  isComparing = false,
  onClick,
  viewMode = 'grid',
}: LLMModelCardProps) {
  const queryClient = useQueryClient()
  const [installing, setInstalling] = useState(false)

  const handleInstallToggle = async (e: React.MouseEvent) => {
    e.stopPropagation()
    setInstalling(true)
    try {
      if (model.is_installed) {
        await apiClient.post(routeActionPath(model, 'uninstall'))
        toast.success(`${model.route_label || model.display_name} removed`)
      } else {
        await apiClient.post(routeActionPath(model, 'install'))
        toast.success(`${model.route_label || model.display_name} installed`)
      }
      queryClient.invalidateQueries({ queryKey: ['marketplaceLlmModels'] })
      queryClient.invalidateQueries({ queryKey: ['workspace-models'] })
      queryClient.invalidateQueries({ queryKey: ['installed-model-ids'] })
      onInstall?.(model.model_id)
    } catch (error: any) {
      toast.error(model.is_installed ? 'Failed to uninstall' : 'Failed to install', {
        description: error?.message || 'An error occurred',
      })
    } finally {
      setInstalling(false)
    }
  }

  const handleCompare = (e: React.MouseEvent) => {
    e.stopPropagation()
    onCompareToggle?.(model.model_id)
  }

  // Top 3 capabilities to show on card
  const capEntries = Object.entries(model.capabilities || {}).slice(0, 3)

  if (viewMode === 'list') {
    return (
      <Card
        className="glass-card card-glow hover:border-primary/20 transition-all cursor-pointer"
        onClick={onClick}
      >
        <CardContent className="p-3">
          <div className="flex items-center gap-3">
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <span className="font-semibold text-sm truncate">{model.display_name}</span>
                <Badge
                  variant="outline"
                  className={`text-[9px] uppercase font-bold tracking-wider shrink-0 ${providerBadgeClass(routeBadges(model).routeKey)}`}
                >
                  {routeBadges(model).route}
                </Badge>
                {routeBadges(model).vendor && (
                  <span className="text-[10px] text-muted-foreground shrink-0">{routeBadges(model).vendor}</span>
                )}
              </div>
              <div className="flex items-center gap-2 text-xs text-muted-foreground mt-0.5">
                <span>{formatTokenCount(model.context_window)} ctx</span>
                <span>&middot;</span>
                <span>{model.is_free ? 'Free' : `In: ${formatCostPer1M(model.input_cost_per_1k)}/1M`}</span>
                {model.key_available === false && (
                  <span className="text-[10px] text-warning">· add a {routeBadges(model).route} key</span>
                )}
              </div>
            </div>
            {model.is_installed ? (
              <Button size="sm" variant="secondary"
                className="h-7 text-xs bg-secondary/50 hover:bg-secondary border border-white/10 shrink-0"
                disabled={installing}
                onClick={handleInstallToggle}
              >
                {installing ? <Loader2 className="w-3 h-3 animate-spin" /> : <CheckCircle className="w-3 h-3" />}
              </Button>
            ) : (
              <Button size="sm" variant="outline"
                className="h-7 text-xs shrink-0"
                disabled={installing}
                onClick={handleInstallToggle}
              >
                {installing ? <Loader2 className="w-3 h-3 animate-spin" /> : <Download className="w-3 h-3" />}
              </Button>
            )}
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card
      className="glass-card card-glow hover:border-primary/20 transition-all duration-300 cursor-pointer flex flex-col relative"
      onClick={onClick}
    >
      {/* Featured star */}
      {model.is_featured && (
        <div className="absolute top-3 right-3 z-10">
          <Star className="w-5 h-5 fill-[hsl(var(--primary))] text-[hsl(var(--primary))]" />
        </div>
      )}

      <CardHeader className="pb-3">
        <div className="flex items-start gap-3">
          {/* Route badge — who serves the call (PRD-236 W1) */}
          <Badge
            variant="outline"
            className={`text-[10px] uppercase font-bold tracking-wider shrink-0 ${providerBadgeClass(routeBadges(model).routeKey)}`}
          >
            {routeBadges(model).route}
          </Badge>

          <div className="flex-1 min-w-0">
            <h3 className="font-semibold text-foreground line-clamp-1 text-sm">
              {model.display_name}
            </h3>
            <p className="text-[11px] text-muted-foreground font-mono truncate">
              {model.model_id}
            </p>
          </div>
        </div>

        {/* Tier + Default + Plan badges */}
        <div className="flex flex-wrap gap-1.5 mt-2">
          {routeBadges(model).vendor && (
            <Badge variant="outline" className="text-[10px] border-border/60 text-muted-foreground">
              by {routeBadges(model).vendor}
            </Badge>
          )}
          {sourcingLabel(model) && (
            <Badge
              variant="outline"
              className={
                (model.sourcing ?? model.tier) === 'direct'
                  ? 'text-[10px] border-[hsl(var(--success))]/30 text-[hsl(var(--success))]'
                  : 'text-[10px] border-[hsl(var(--info))]/30 text-[hsl(var(--info))]'
              }
            >
              {sourcingLabel(model)}
            </Badge>
          )}
          {model.is_free && (
            <Badge variant="outline" className="text-[10px] border-lime-500/40 text-lime-400">
              Free
            </Badge>
          )}
          {model.key_available === false && (
            <Badge variant="outline" className="text-[10px] border-warning/40 text-warning">
              Add a {routeBadges(model).route} key
            </Badge>
          )}
          {model.is_default && (
            <Badge
              variant="outline"
              className="text-[10px] border-[hsl(var(--agent))]/30 text-[hsl(var(--agent))]"
            >
              Default
            </Badge>
          )}
          {model.requires_plan && (
            <Badge
              variant="outline"
              className="text-[10px] border-[hsl(var(--primary))]/30 text-[hsl(var(--primary))]"
            >
              <Shield className="w-2.5 h-2.5 mr-0.5" />
              {model.requires_plan}
            </Badge>
          )}
        </div>
      </CardHeader>

      <CardContent className="flex-1 flex flex-col space-y-3 pt-0">
        {/* Description */}
        {model.description && (
          <p className="text-xs text-muted-foreground line-clamp-2">
            {model.description}
          </p>
        )}

        {/* Context & Output */}
        <div className="grid grid-cols-2 gap-2">
          <div className="rounded-md border border-border/40 bg-secondary/20 p-2 text-center">
            <div className="text-sm font-bold text-foreground">
              {formatTokenCount(model.context_window)}
            </div>
            <div className="text-[10px] text-muted-foreground">Context</div>
          </div>
          <div className="rounded-md border border-border/40 bg-secondary/20 p-2 text-center">
            <div className="text-sm font-bold text-foreground">
              {formatTokenCount(model.max_output_tokens)}
            </div>
            <div className="text-[10px] text-muted-foreground">Max Output</div>
          </div>
        </div>

        {/* Cost per 1M tokens — per ROUTE (PRD-236 W1) */}
        <div className="flex items-center justify-between text-xs">
          <span className="text-muted-foreground">Cost / 1M tokens</span>
          {model.is_free ? (
            <span className="text-lime-400 font-medium" title={model.terms_note || undefined}>
              Free{model.rate_limit_note ? ' · rate-limited' : ''}
            </span>
          ) : (
            <div className="flex items-center gap-2">
              <span className="text-[hsl(var(--success))] font-medium">
                In: {formatCostPer1M(model.input_cost_per_1k)}
              </span>
              <span className="text-muted-foreground">/</span>
              <span className="text-[hsl(var(--primary))] font-medium">
                Out: {formatCostPer1M(model.output_cost_per_1k)}
              </span>
            </div>
          )}
        </div>

        {/* Capability bars */}
        {capEntries.length > 0 && (
          <div className="space-y-2">
            {capEntries.map(([capName, rating]) => {
              const r = CAPABILITY_RATING[rating] || CAPABILITY_RATING.moderate
              return (
                <div key={capName} className="flex items-center gap-2">
                  <span className="text-[10px] text-muted-foreground w-16 truncate capitalize">
                    {capName}
                  </span>
                  <div className="flex-1 h-1.5 rounded-full bg-secondary/50">
                    <div className={`h-full rounded-full ${r.color} ${r.width}`} />
                  </div>
                </div>
              )
            })}
          </div>
        )}

        {/* Feature badges */}
        <div className="flex flex-wrap gap-1.5">
          {model.supports_functions && (
            <Badge variant="outline" className="text-[10px] border-[hsl(var(--info))]/20 text-[hsl(var(--info))]">
              <Braces className="w-2.5 h-2.5 mr-0.5" />
              Functions
            </Badge>
          )}
          {model.supports_vision && (
            <Badge variant="outline" className="text-[10px] border-[hsl(var(--agent))]/20 text-[hsl(var(--agent))]">
              <Eye className="w-2.5 h-2.5 mr-0.5" />
              Vision
            </Badge>
          )}
          {model.supports_streaming && (
            <Badge variant="outline" className="text-[10px] border-[hsl(var(--success))]/20 text-[hsl(var(--success))]">
              <Radio className="w-2.5 h-2.5 mr-0.5" />
              Streaming
            </Badge>
          )}
        </div>

        {/* Recommended for tags */}
        {model.recommended_for && model.recommended_for.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {model.recommended_for.slice(0, 3).map((rec) => (
              <Badge
                key={rec}
                variant="outline"
                className="text-[10px] border-border text-muted-foreground"
              >
                {rec}
              </Badge>
            ))}
            {model.recommended_for.length > 3 && (
              <Badge variant="outline" className="text-[10px] border-border text-muted-foreground">
                +{model.recommended_for.length - 3}
              </Badge>
            )}
          </div>
        )}

        {/* Footer: install count, compare, install button */}
        <div className="pt-3 mt-auto border-t border-border/40 flex items-center justify-between gap-2">
          <div className="flex items-center gap-3">
            <span className="text-[11px] text-muted-foreground flex items-center gap-1">
              <Download className="w-3 h-3" />
              {formatInstallCount(model.install_count)}
            </span>
            {onCompareToggle && (
              <button
                onClick={handleCompare}
                className="flex items-center gap-1 text-[11px] text-muted-foreground hover:text-foreground transition-colors"
              >
                <ArrowRightLeft className="w-3 h-3" />
                {isComparing ? 'Comparing' : 'Compare'}
              </button>
            )}
          </div>

          {model.is_installed ? (
            <Button size="sm" variant="secondary"
              className="h-7 text-xs bg-secondary/50 hover:bg-secondary border border-white/10"
              disabled={installing}
              onClick={handleInstallToggle}
            >
              {installing ? (
                <Loader2 className="w-3 h-3 animate-spin" />
              ) : (
                <>
                  <CheckCircle className="w-3 h-3 mr-1" />
                  Added
                </>
              )}
            </Button>
          ) : (
            <Button size="sm" variant="outline"
              className="h-7 text-xs"
              disabled={installing}
              onClick={handleInstallToggle}
            >
              {installing ? (
                <Loader2 className="w-3 h-3 animate-spin" />
              ) : (
                'Add to Workspace'
              )}
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
