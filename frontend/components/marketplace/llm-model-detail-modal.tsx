'use client'

import { useState, useMemo } from 'react'
import {
  Download,
  Loader2,
  CheckCircle,
  Star,
  Shield,
  Braces,
  Eye,
  Radio,
  FileJson,
  Cpu,
  Calculator,
  Tag,
  Sparkles,
} from 'lucide-react'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from '@/components/ui/dialog'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { apiClient } from '@/lib/api-client'
import { useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'
import type { LLMModel } from './llm-model-card'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface LLMModelDetailModalProps {
  model: LLMModel | null
  open: boolean
  onOpenChange: (open: boolean) => void
  onInstall?: (modelId: string) => void
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const PROVIDER_COLORS: Record<string, string> = {
  openai: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30',
  anthropic: 'bg-orange-500/15 text-orange-400 border-orange-500/30',
  google: 'bg-blue-500/15 text-blue-400 border-blue-500/30',
  openrouter: 'bg-purple-500/15 text-purple-400 border-purple-500/30',
  meta: 'bg-sky-500/15 text-sky-400 border-sky-500/30',
  mistral: 'bg-amber-500/15 text-amber-400 border-amber-500/30',
  cohere: 'bg-pink-500/15 text-pink-400 border-pink-500/30',
}

const CAPABILITY_RATING: Record<string, { label: string; pct: number; color: string }> = {
  excellent: { label: 'Excellent', pct: 100, color: 'bg-[hsl(var(--success))]' },
  good: { label: 'Good', pct: 75, color: 'bg-[hsl(var(--info))]' },
  moderate: { label: 'Moderate', pct: 50, color: 'bg-[hsl(var(--primary))]' },
  basic: { label: 'Basic', pct: 25, color: 'bg-muted-foreground' },
  none: { label: 'None', pct: 0, color: 'bg-muted' },
}

function formatTokenCount(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(n % 1_000_000 === 0 ? 0 : 1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(n % 1_000 === 0 ? 0 : 1)}K`
  return n.toString()
}

function formatCost(costPer1K: number, label: string): { per1K: string; per1M: string } {
  const per1M = costPer1K * 1000
  const fmt = (v: number) => {
    if (v === 0) return 'Free'
    if (v < 0.01) return `$${v.toFixed(4)}`
    if (v < 1) return `$${v.toFixed(3)}`
    return `$${v.toFixed(2)}`
  }
  return { per1K: fmt(costPer1K), per1M: fmt(per1M) }
}

function providerBadgeClass(provider: string): string {
  return PROVIDER_COLORS[provider.toLowerCase()] || 'bg-secondary text-muted-foreground border-border'
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function LLMModelDetailModal({
  model,
  open,
  onOpenChange,
  onInstall,
}: LLMModelDetailModalProps) {
  const queryClient = useQueryClient()
  const [installing, setInstalling] = useState(false)

  // Cost calculator state
  const [inputTokensK, setInputTokensK] = useState<string>('100')
  const [outputTokensK, setOutputTokensK] = useState<string>('50')

  const monthlyCost = useMemo(() => {
    if (!model) return 0
    const inK = parseFloat(inputTokensK) || 0
    const outK = parseFloat(outputTokensK) || 0
    return inK * model.input_cost_per_1k + outK * model.output_cost_per_1k
  }, [model, inputTokensK, outputTokensK])

  if (!model) return null

  const handleInstallToggle = async () => {
    setInstalling(true)
    try {
      if (model.is_installed) {
        await apiClient.post(`/api/marketplace/llm/models/${model.model_id}/uninstall`)
        toast.success(`${model.display_name} removed`)
      } else {
        await apiClient.post(`/api/marketplace/llm/models/${model.model_id}/install`)
        toast.success(`${model.display_name} installed`)
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

  const inputCost = formatCost(model.input_cost_per_1k, 'Input')
  const outputCost = formatCost(model.output_cost_per_1k, 'Output')
  const capEntries = Object.entries(model.capabilities || {})

  const featureList = [
    { key: 'functions', label: 'Function Calling', supported: model.supports_functions, icon: Braces, color: 'info' },
    { key: 'vision', label: 'Vision', supported: model.supports_vision, icon: Eye, color: 'agent' },
    { key: 'streaming', label: 'Streaming', supported: model.supports_streaming, icon: Radio, color: 'success' },
  ]

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <div className="flex items-start gap-3">
            <Badge
              variant="outline"
              className={`text-[10px] uppercase font-bold tracking-wider shrink-0 mt-1 ${providerBadgeClass(model.provider)}`}
            >
              {model.provider}
            </Badge>
            <div className="flex-1 min-w-0">
              <DialogTitle className="flex items-center gap-2 flex-wrap">
                <span>{model.display_name}</span>
                {model.is_featured && (
                  <Star className="w-4 h-4 fill-[hsl(var(--primary))] text-[hsl(var(--primary))]" />
                )}
              </DialogTitle>
              <DialogDescription className="font-mono text-xs mt-1">
                {model.model_id}
              </DialogDescription>
            </div>
          </div>

          {/* Tier + Default + Plan badges */}
          <div className="flex flex-wrap gap-1.5 mt-3">
            {model.tier && (
              <Badge
                variant="outline"
                className={
                  model.tier === 'direct'
                    ? 'text-[10px] border-[hsl(var(--success))]/30 text-[hsl(var(--success))]'
                    : 'text-[10px] border-[hsl(var(--info))]/30 text-[hsl(var(--info))]'
                }
              >
                {model.tier === 'direct' ? 'Direct API' : 'Aggregator'}
              </Badge>
            )}
            {model.is_default && (
              <Badge variant="outline" className="text-[10px] border-[hsl(var(--agent))]/30 text-[hsl(var(--agent))]">
                Default Model
              </Badge>
            )}
            {model.requires_plan && (
              <Badge variant="outline" className="text-[10px] border-[hsl(var(--primary))]/30 text-[hsl(var(--primary))]">
                <Shield className="w-2.5 h-2.5 mr-0.5" />
                Requires {model.requires_plan}
              </Badge>
            )}
            {model.category && (
              <Badge variant="outline" className="text-[10px] border-border text-muted-foreground">
                {model.category}
              </Badge>
            )}
          </div>
        </DialogHeader>

        <div className="space-y-6 mt-2">
          {/* Description */}
          {model.description && (
            <div className="rounded-lg border border-border/40 bg-secondary/20 p-4">
              <p className="text-sm text-muted-foreground leading-relaxed">
                {model.description}
              </p>
            </div>
          )}

          {/* Specifications */}
          <div className="space-y-3">
            <h3 className="text-sm font-semibold flex items-center gap-2">
              <Cpu className="w-4 h-4" />
              Specifications
            </h3>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <div className="rounded-lg border border-border/40 bg-secondary/20 p-3 text-center">
                <div className="text-lg font-bold text-foreground">
                  {formatTokenCount(model.context_window)}
                </div>
                <div className="text-[11px] text-muted-foreground">Context Window</div>
              </div>
              <div className="rounded-lg border border-border/40 bg-secondary/20 p-3 text-center">
                <div className="text-lg font-bold text-foreground">
                  {formatTokenCount(model.max_output_tokens)}
                </div>
                <div className="text-[11px] text-muted-foreground">Max Output</div>
              </div>
              <div className="rounded-lg border border-border/40 bg-secondary/20 p-3 text-center">
                <div className="text-lg font-bold text-[hsl(var(--success))]">
                  {inputCost.per1K}
                </div>
                <div className="text-[11px] text-muted-foreground">Input / 1K</div>
              </div>
              <div className="rounded-lg border border-border/40 bg-secondary/20 p-3 text-center">
                <div className="text-lg font-bold text-[hsl(var(--primary))]">
                  {outputCost.per1K}
                </div>
                <div className="text-[11px] text-muted-foreground">Output / 1K</div>
              </div>
            </div>

            {/* Per 1M summary */}
            <div className="flex items-center justify-center gap-4 text-xs text-muted-foreground">
              <span>
                Per 1M tokens: Input{' '}
                <span className="text-[hsl(var(--success))] font-medium">{inputCost.per1M}</span>
              </span>
              <span className="text-border">|</span>
              <span>
                Output{' '}
                <span className="text-[hsl(var(--primary))] font-medium">{outputCost.per1M}</span>
              </span>
            </div>
          </div>

          {/* Capabilities */}
          {capEntries.length > 0 && (
            <div className="space-y-3">
              <h3 className="text-sm font-semibold flex items-center gap-2">
                <Sparkles className="w-4 h-4" />
                Capabilities
              </h3>
              <div className="space-y-2">
                {capEntries.map(([capName, rating]) => {
                  const r = CAPABILITY_RATING[rating] || CAPABILITY_RATING.moderate
                  return (
                    <div key={capName} className="flex items-center gap-3">
                      <span className="text-xs text-muted-foreground w-24 capitalize truncate">
                        {capName}
                      </span>
                      <div className="flex-1 h-2 rounded-full bg-secondary/50">
                        <div
                          className={`h-full rounded-full transition-all ${r.color}`}
                          style={{ width: `${r.pct}%` }}
                        />
                      </div>
                      <span className="text-[11px] text-muted-foreground w-16 text-right">
                        {r.label}
                      </span>
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          {/* Features */}
          <div className="space-y-3">
            <h3 className="text-sm font-semibold flex items-center gap-2">
              <FileJson className="w-4 h-4" />
              Features
            </h3>
            <div className="grid grid-cols-3 gap-2">
              {featureList.map((feat) => {
                const Icon = feat.icon
                return (
                  <div
                    key={feat.key}
                    className={`rounded-lg border p-3 text-center transition-colors ${
                      feat.supported
                        ? `border-[hsl(var(--${feat.color}))]/30 bg-[hsl(var(--${feat.color}))]/5`
                        : 'border-border/30 bg-secondary/10 opacity-40'
                    }`}
                  >
                    <Icon
                      className={`w-5 h-5 mx-auto mb-1 ${
                        feat.supported ? `text-[hsl(var(--${feat.color}))]` : 'text-muted-foreground'
                      }`}
                    />
                    <div className="text-[11px] font-medium">
                      {feat.label}
                    </div>
                    <div className={`text-[10px] mt-0.5 ${feat.supported ? 'text-[hsl(var(--success))]' : 'text-muted-foreground'}`}>
                      {feat.supported ? 'Supported' : 'Not supported'}
                    </div>
                  </div>
                )
              })}
            </div>
          </div>

          {/* Recommended For */}
          {model.recommended_for && model.recommended_for.length > 0 && (
            <div className="space-y-3">
              <h3 className="text-sm font-semibold flex items-center gap-2">
                <Tag className="w-4 h-4" />
                Recommended For
              </h3>
              <div className="flex flex-wrap gap-2">
                {model.recommended_for.map((rec) => (
                  <Badge
                    key={rec}
                    variant="outline"
                    className="text-xs border-[hsl(var(--info))]/30 text-[hsl(var(--info))]"
                  >
                    {rec}
                  </Badge>
                ))}
              </div>
            </div>
          )}

          {/* Tags */}
          {model.tags && model.tags.length > 0 && (
            <div className="flex flex-wrap gap-1.5">
              {model.tags.map((tag) => (
                <Badge
                  key={tag}
                  variant="outline"
                  className="text-[10px] border-border text-muted-foreground"
                >
                  {tag}
                </Badge>
              ))}
            </div>
          )}

          {/* Cost Calculator */}
          <div className="space-y-3">
            <h3 className="text-sm font-semibold flex items-center gap-2">
              <Calculator className="w-4 h-4" />
              Cost Calculator
            </h3>
            <div className="rounded-lg border border-border/40 bg-secondary/20 p-4 space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <label className="text-xs text-muted-foreground">
                    Input tokens (thousands / month)
                  </label>
                  <Input
                    type="number"
                    min="0"
                    step="10"
                    value={inputTokensK}
                    onChange={(e) => setInputTokensK(e.target.value)}
                    className="h-8 text-sm"
                    onClick={(e) => e.stopPropagation()}
                  />
                </div>
                <div className="space-y-1.5">
                  <label className="text-xs text-muted-foreground">
                    Output tokens (thousands / month)
                  </label>
                  <Input
                    type="number"
                    min="0"
                    step="10"
                    value={outputTokensK}
                    onChange={(e) => setOutputTokensK(e.target.value)}
                    className="h-8 text-sm"
                    onClick={(e) => e.stopPropagation()}
                  />
                </div>
              </div>
              <div className="flex items-center justify-between pt-3 border-t border-border/30">
                <span className="text-sm text-muted-foreground">Estimated monthly cost</span>
                <span className="text-lg font-bold text-foreground">
                  ${monthlyCost.toFixed(4)}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <DialogFooter className="mt-4">
          <div className="flex items-center justify-between w-full gap-3">
            <span className="text-xs text-muted-foreground flex items-center gap-1">
              <Download className="w-3.5 h-3.5" />
              {model.install_count.toLocaleString()} installs
            </span>

            <Button
              variant={model.is_installed ? 'secondary' : 'default'}
              className={
                model.is_installed
                  ? 'bg-[hsl(var(--success))]/10 text-[hsl(var(--success))] border border-[hsl(var(--success))]/30 hover:bg-[hsl(var(--destructive))]/10 hover:text-[hsl(var(--destructive))] hover:border-[hsl(var(--destructive))]/30'
                  : ''
              }
              disabled={installing}
              onClick={handleInstallToggle}
            >
              {installing ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  {model.is_installed ? 'Removing...' : 'Installing...'}
                </>
              ) : model.is_installed ? (
                <>
                  <CheckCircle className="w-4 h-4 mr-2" />
                  Installed — Click to Remove
                </>
              ) : (
                <>
                  <Download className="w-4 h-4 mr-2" />
                  Install Model
                </>
              )}
            </Button>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
