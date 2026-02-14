/**
 * Model Selector Component (PRD-15, PRD-54)
 * ==========================================
 *
 * React component for selecting and displaying LLM models.
 * Groups models by tier, shows inline cost/context, and model details.
 */

'use client'

import { useState, useEffect, useMemo } from 'react'
import { motion } from 'framer-motion'
import {
  Brain,
  Zap,
  DollarSign,
  Info,
  Check,
  AlertCircle,
  Sparkles,
  Star,
  Globe,
  Key
} from 'lucide-react'
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'
import { Skeleton } from '@/components/ui/skeleton'
import { useModels, ModelInfo } from '@/hooks/use-model-api'

interface ModelSelectorProps {
  value: string  // model_id
  onChange: (modelId: string) => void
  agentType?: string
  provider?: string
  className?: string
}

// Tier ordering and display config
const TIER_CONFIG: Record<string, { label: string; icon: typeof Star; order: number }> = {
  direct: { label: 'Direct Providers', icon: Star, order: 1 },
  openrouter: { label: 'OpenRouter', icon: Globe, order: 2 },
  byok: { label: 'Your Keys (BYOK)', icon: Key, order: 3 },
}

export function ModelSelector({
  value,
  onChange,
  agentType,
  provider,
  className = ''
}: ModelSelectorProps) {
  const [selectedModel, setSelectedModel] = useState<ModelInfo | null>(null)
  const { data: models, isLoading, error } = useModels(provider)

  useEffect(() => {
    if (value && models && models.length > 0) {
      const model = models.find(m => m.model_id === value)
      setSelectedModel(model || null)
    }
  }, [value, models])

  const handleChange = (newValue: string) => {
    onChange(newValue)
    const model = models?.find(m => m.model_id === newValue)
    setSelectedModel(model || null)
  }

  // Group models by tier, then by provider within each tier
  const tieredModels = useMemo(() => {
    if (!models) return null

    const tiers: Record<string, Record<string, ModelInfo[]>> = {}

    for (const model of models) {
      const tier = model.tier || 'direct'
      if (!tiers[tier]) tiers[tier] = {}
      if (!tiers[tier][model.provider]) tiers[tier][model.provider] = []
      tiers[tier][model.provider].push(model)
    }

    // Sort tiers by configured order
    const sorted = Object.entries(tiers).sort(([a], [b]) => {
      const orderA = TIER_CONFIG[a]?.order ?? 99
      const orderB = TIER_CONFIG[b]?.order ?? 99
      return orderA - orderB
    })

    // Put recommended models (matching agentType) at top if agentType given
    if (agentType) {
      const recommended = models.filter(m => m.recommended_for?.includes(agentType))
      if (recommended.length > 0) {
        return { recommended, tiers: sorted }
      }
    }

    return { recommended: [] as ModelInfo[], tiers: sorted }
  }, [models, agentType])

  const getProviderColor = (provider: string) => {
    const colors: Record<string, string> = {
      openai: 'bg-[hsl(var(--success))]/10 text-[hsl(var(--success))] border-[hsl(var(--success))]/20',
      anthropic: 'bg-[hsl(var(--agent))]/10 text-[hsl(var(--agent))] border-[hsl(var(--agent))]/20',
      google: 'bg-[hsl(var(--info))]/10 text-[hsl(var(--info))] border-[hsl(var(--info))]/20',
      azure: 'bg-[hsl(var(--info))]/10 text-[hsl(var(--info))] border-[hsl(var(--info))]/20',
      huggingface: 'bg-[hsl(var(--warning))]/10 text-[hsl(var(--warning))] border-[hsl(var(--warning))]/20',
      aws_bedrock: 'bg-primary/10 text-primary border-primary/20',
      bedrock: 'bg-primary/10 text-primary border-primary/20',
      openrouter: 'bg-[hsl(var(--warning))]/10 text-[hsl(var(--warning))] border-[hsl(var(--warning))]/20',
      grok: 'bg-[hsl(var(--destructive))]/10 text-[hsl(var(--destructive))] border-[hsl(var(--destructive))]/20'
    }
    return colors[provider.toLowerCase()] || 'bg-secondary/50 text-muted-foreground border-border/50'
  }

  const getTierColor = (tier: string) => {
    const colors: Record<string, string> = {
      direct: 'text-[hsl(var(--success))]',
      openrouter: 'text-[hsl(var(--warning))]',
      byok: 'text-[hsl(var(--info))]',
    }
    return colors[tier] || 'text-muted-foreground'
  }

  const getCapabilityBadge = (capability: string, rating: string) => {
    const isExcellent = rating.toLowerCase().includes('excellent')
    const isGood = rating.toLowerCase().includes('good')
    const isFast = rating.toLowerCase().includes('fast')

    let colorClass = 'border-border/50 text-muted-foreground'
    if (isExcellent) colorClass = 'border-[hsl(var(--success))]/30 text-[hsl(var(--success))]'
    else if (isGood) colorClass = 'border-[hsl(var(--info))]/30 text-[hsl(var(--info))]'
    else if (isFast) colorClass = 'border-[hsl(var(--warning))]/30 text-[hsl(var(--warning))]'

    return (
      <Badge
        key={capability}
        variant="outline"
        className={`text-xs ${colorClass}`}
      >
        {capability}: {rating}
      </Badge>
    )
  }

  const formatNumber = (num: number) => {
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`
    if (num >= 1000) return `${(num / 1000).toFixed(0)}K`
    return num.toString()
  }

  const formatCost = (cost: number) => {
    if (cost === 0) return 'Free'
    if (cost < 0.001) return `$${cost.toFixed(5)}`
    return `$${cost.toFixed(4)}`
  }

  const renderModelItem = (model: ModelInfo) => (
    <SelectItem
      key={model.model_id}
      value={model.model_id}
      className="text-foreground focus:bg-secondary focus:text-foreground py-2"
    >
      <div className="flex items-center gap-2 w-full">
        <span className="truncate">{model.display_name}</span>
        <span className="text-[10px] text-muted-foreground ml-auto shrink-0">
          {formatNumber(model.context_window)}ctx
        </span>
        <span className="text-[10px] text-muted-foreground shrink-0">
          {formatCost(model.input_cost_per_1k)}/1K
        </span>
        {model.status === 'beta' && (
          <Badge variant="outline" className="text-[10px] h-4 px-1">Beta</Badge>
        )}
        {model.is_featured && (
          <Star className="h-3 w-3 text-[hsl(var(--warning))] shrink-0" />
        )}
        {agentType && model.recommended_for?.includes(agentType) && (
          <Sparkles className="h-3 w-3 text-[hsl(var(--warning))] shrink-0" />
        )}
      </div>
    </SelectItem>
  )

  if (error) {
    return (
      <div className="p-4 rounded-lg border border-[hsl(var(--destructive))]/20 bg-[hsl(var(--destructive))]/10">
        <div className="flex items-center gap-2 text-[hsl(var(--destructive))] text-sm">
          <AlertCircle className="h-4 w-4" />
          <span>Failed to load models. Please try again.</span>
        </div>
      </div>
    )
  }

  return (
    <div className={`space-y-4 ${className}`}>
      {/* Model Selector */}
      <div className="space-y-2">
        <label className="text-sm font-medium text-foreground">Model</label>
        <Select value={value} onValueChange={handleChange} disabled={isLoading}>
          <SelectTrigger className="w-full bg-secondary/30 border-border/50">
            <SelectValue placeholder={isLoading ? "Loading models..." : "Select a model..."} />
          </SelectTrigger>
          <SelectContent className="bg-popover/95 border-border/50 max-h-[400px]">
            {isLoading ? (
              <div className="p-4 space-y-2">
                <Skeleton className="h-4 w-full" />
                <Skeleton className="h-4 w-3/4" />
              </div>
            ) : tieredModels && (
              <>
                {/* Recommended models section */}
                {tieredModels.recommended.length > 0 && (
                  <SelectGroup>
                    <SelectLabel className="text-xs uppercase px-2 py-1.5 flex items-center gap-1">
                      <Sparkles className="h-3 w-3 text-[hsl(var(--warning))]" />
                      <span className="text-[hsl(var(--warning))]">Recommended for {agentType}</span>
                    </SelectLabel>
                    {tieredModels.recommended.map(renderModelItem)}
                  </SelectGroup>
                )}

                {/* Tier-grouped models */}
                {tieredModels.tiers.map(([tier, providers]) => (
                  <div key={tier}>
                    <SelectGroup>
                      <SelectLabel className={`text-xs uppercase px-2 py-1.5 flex items-center gap-1 ${getTierColor(tier)}`}>
                        {TIER_CONFIG[tier] ? (
                          <>
                            {(() => {
                              const Icon = TIER_CONFIG[tier].icon
                              return <Icon className="h-3 w-3" />
                            })()}
                            <span>{TIER_CONFIG[tier].label}</span>
                          </>
                        ) : (
                          <span>{tier}</span>
                        )}
                      </SelectLabel>
                    </SelectGroup>
                    {Object.entries(providers).map(([providerName, providerModels]) => (
                      <SelectGroup key={`${tier}-${providerName}`}>
                        <SelectLabel className="text-[10px] uppercase text-muted-foreground/60 px-4 py-0.5">
                          {providerName}
                        </SelectLabel>
                        {providerModels.map(renderModelItem)}
                      </SelectGroup>
                    ))}
                  </div>
                ))}
              </>
            )}
          </SelectContent>
        </Select>
      </div>

      {/* Model Details Card */}
      {selectedModel && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.2 }}
        >
          <Card className="bg-secondary/30 border-border/50">
            <CardContent className="p-4 space-y-4">
              {/* Header */}
              <div className="flex items-start justify-between">
                <div>
                  <h4 className="font-semibold text-foreground">
                    {selectedModel.display_name}
                  </h4>
                  <p className="text-sm text-muted-foreground">
                    {selectedModel.model_family} by {selectedModel.provider}
                  </p>
                </div>
                <div className="flex items-center gap-1.5">
                  {selectedModel.tier && selectedModel.tier !== 'direct' && (
                    <Badge variant="outline" className={`text-xs ${getTierColor(selectedModel.tier)}`}>
                      {selectedModel.tier}
                    </Badge>
                  )}
                  <Badge className={getProviderColor(selectedModel.provider)}>
                    {selectedModel.provider}
                  </Badge>
                </div>
              </div>

              {/* Key Metrics */}
              <div className="grid grid-cols-3 gap-3">
                <div className="space-y-1">
                  <div className="flex items-center gap-1 text-xs text-muted-foreground">
                    <Brain className="h-3 w-3" />
                    <span>Context</span>
                  </div>
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <p className="text-sm font-medium text-foreground cursor-help">
                          {formatNumber(selectedModel.context_window)} tokens
                        </p>
                      </TooltipTrigger>
                      <TooltipContent>
                        <p className="text-xs">Maximum input context: {selectedModel.context_window.toLocaleString()} tokens</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>

                <div className="space-y-1">
                  <div className="flex items-center gap-1 text-xs text-muted-foreground">
                    <Zap className="h-3 w-3" />
                    <span>Max Output</span>
                  </div>
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <p className="text-sm font-medium text-foreground cursor-help">
                          {formatNumber(selectedModel.max_output_tokens)} tokens
                        </p>
                      </TooltipTrigger>
                      <TooltipContent>
                        <p className="text-xs">Maximum output: {selectedModel.max_output_tokens.toLocaleString()} tokens</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>

                <div className="space-y-1">
                  <div className="flex items-center gap-1 text-xs text-muted-foreground">
                    <DollarSign className="h-3 w-3" />
                    <span>Cost</span>
                  </div>
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <p className="text-sm font-medium text-foreground cursor-help">
                          {formatCost(selectedModel.input_cost_per_1k)}/1K
                        </p>
                      </TooltipTrigger>
                      <TooltipContent>
                        <div className="text-xs space-y-1">
                          <div>Input: {formatCost(selectedModel.input_cost_per_1k)}/1K tokens</div>
                          <div>Output: {formatCost(selectedModel.output_cost_per_1k)}/1K tokens</div>
                        </div>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>
              </div>

              {/* Capabilities */}
              {Object.keys(selectedModel.capabilities).length > 0 && (
                <div className="space-y-2">
                  <div className="flex items-center gap-1 text-xs text-muted-foreground">
                    <Info className="h-3 w-3" />
                    <span>Capabilities</span>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {Object.entries(selectedModel.capabilities).map(([cap, rating]) =>
                      getCapabilityBadge(cap, rating as string)
                    )}
                  </div>
                </div>
              )}

              {/* Features */}
              <div className="flex flex-wrap gap-2">
                {selectedModel.supports_functions && (
                  <Badge variant="outline" className="border-[hsl(var(--info))]/30 text-[hsl(var(--info))] text-xs">
                    <Check className="h-3 w-3 mr-1" />
                    Function Calling
                  </Badge>
                )}
                {selectedModel.supports_vision && (
                  <Badge variant="outline" className="border-[hsl(var(--agent))]/30 text-[hsl(var(--agent))] text-xs">
                    <Check className="h-3 w-3 mr-1" />
                    Vision
                  </Badge>
                )}
                {selectedModel.supports_streaming && (
                  <Badge variant="outline" className="border-border/50 text-muted-foreground text-xs">
                    <Check className="h-3 w-3 mr-1" />
                    Streaming
                  </Badge>
                )}
              </div>

              {/* Recommended For */}
              {selectedModel.recommended_for && selectedModel.recommended_for.length > 0 && (
                <div className="space-y-2">
                  <div className="text-xs text-muted-foreground">Recommended for:</div>
                  <div className="flex flex-wrap gap-1">
                    {selectedModel.recommended_for.map((task) => (
                      <Badge
                        key={task}
                        variant="secondary"
                        className="text-xs bg-secondary/50 text-muted-foreground"
                      >
                        {task.replace(/_/g, ' ')}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}

              {/* Description */}
              {selectedModel.description && (
                <p className="text-xs text-muted-foreground leading-relaxed">
                  {selectedModel.description}
                </p>
              )}

              {/* Warning for deprecated */}
              {selectedModel.status === 'deprecated' && (
                <div className="flex items-start gap-2 p-2 rounded-lg bg-[hsl(var(--warning))]/10 border border-[hsl(var(--warning))]/20">
                  <AlertCircle className="h-4 w-4 text-[hsl(var(--warning))] mt-0.5 flex-shrink-0" />
                  <div className="text-xs text-[hsl(var(--warning))]">
                    This model is deprecated and may be removed in the future.
                  </div>
                </div>
              )}

              {/* Beta warning */}
              {selectedModel.status === 'beta' && (
                <div className="flex items-start gap-2 p-2 rounded-lg bg-[hsl(var(--info))]/10 border border-[hsl(var(--info))]/20">
                  <Info className="h-4 w-4 text-[hsl(var(--info))] mt-0.5 flex-shrink-0" />
                  <div className="text-xs text-[hsl(var(--info))]">
                    This model is in beta. Features may change.
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </motion.div>
      )}
    </div>
  )
}
