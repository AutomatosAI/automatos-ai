/**
 * Model Selector Component (PRD-15)
 * =================================
 * 
 * React component for selecting and displaying LLM models.
 * Shows model details, capabilities, costs, and recommendations.
 */

'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { 
  Brain, 
  Zap, 
  DollarSign, 
  Info,
  Check,
  AlertCircle,
  Sparkles,
  ChevronDown
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

  const getProviderColor = (provider: string) => {
    const colors: Record<string, string> = {
      openai: 'bg-green-500/10 text-green-400 border-green-500/20',
      anthropic: 'bg-purple-500/10 text-purple-400 border-purple-500/20',
      huggingface: 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20'
    }
    return colors[provider.toLowerCase()] || 'bg-gray-500/10 text-gray-400 border-gray-500/20'
  }

  const getCapabilityBadge = (capability: string, rating: string) => {
    const isExcellent = rating.toLowerCase().includes('excellent')
    const isGood = rating.toLowerCase().includes('good')
    const isFast = rating.toLowerCase().includes('fast')
    
    let colorClass = 'border-gray-500/30 text-gray-400'
    if (isExcellent) colorClass = 'border-green-500/30 text-green-400'
    else if (isGood) colorClass = 'border-blue-500/30 text-blue-400'
    else if (isFast) colorClass = 'border-yellow-500/30 text-yellow-400'
    
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
    if (num >= 1000) return `${(num / 1000).toFixed(0)}K`
    return num.toString()
  }

  // Group models by provider
  const groupedModels = models?.reduce((acc, model) => {
    if (!acc[model.provider]) {
      acc[model.provider] = []
    }
    acc[model.provider].push(model)
    return acc
  }, {} as Record<string, ModelInfo[]>)

  if (error) {
    return (
      <div className="p-4 rounded-lg border border-red-500/20 bg-red-500/10">
        <div className="flex items-center gap-2 text-red-400 text-sm">
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
        <label className="text-sm font-medium text-gray-200">Model</label>
        <Select value={value} onValueChange={handleChange} disabled={isLoading}>
          <SelectTrigger className="w-full bg-black/20 border-gray-800">
            <SelectValue placeholder={isLoading ? "Loading models..." : "Select a model..."} />
          </SelectTrigger>
          <SelectContent className="bg-gray-900 border-gray-800">
            {isLoading ? (
              <div className="p-4">
                <Skeleton className="h-4 w-full" />
              </div>
            ) : (
              groupedModels && Object.entries(groupedModels).map(([provider, providerModels]) => (
                <SelectGroup key={provider}>
                  <SelectLabel className="text-xs uppercase text-gray-400 px-2 py-1.5">
                    {provider}
                  </SelectLabel>
                  {providerModels.map((model) => (
                    <SelectItem 
                      key={model.model_id} 
                      value={model.model_id}
                      className="text-gray-200 focus:bg-gray-800 focus:text-white"
                    >
                      <div className="flex items-center gap-2">
                        <span>{model.display_name}</span>
                        {model.status === 'beta' && (
                          <Badge variant="outline" className="text-xs">Beta</Badge>
                        )}
                        {agentType && model.recommended_for.includes(agentType) && (
                          <TooltipProvider>
                            <Tooltip>
                              <TooltipTrigger>
                                <Sparkles className="h-3 w-3 text-yellow-400" />
                              </TooltipTrigger>
                              <TooltipContent>
                                <p className="text-xs">Recommended for {agentType}</p>
                              </TooltipContent>
                            </Tooltip>
                          </TooltipProvider>
                        )}
                      </div>
                    </SelectItem>
                  ))}
                </SelectGroup>
              ))
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
          <Card className="bg-black/20 border-gray-800">
            <CardContent className="p-4 space-y-4">
              {/* Header */}
              <div className="flex items-start justify-between">
                <div>
                  <h4 className="font-semibold text-white">
                    {selectedModel.display_name}
                  </h4>
                  <p className="text-sm text-gray-400">
                    {selectedModel.model_family} by {selectedModel.provider}
                  </p>
                </div>
                <Badge className={getProviderColor(selectedModel.provider)}>
                  {selectedModel.provider}
                </Badge>
              </div>

              {/* Key Metrics */}
              <div className="grid grid-cols-3 gap-3">
                <div className="space-y-1">
                  <div className="flex items-center gap-1 text-xs text-gray-400">
                    <Brain className="h-3 w-3" />
                    <span>Context</span>
                  </div>
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <p className="text-sm font-medium text-white cursor-help">
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
                  <div className="flex items-center gap-1 text-xs text-gray-400">
                    <Zap className="h-3 w-3" />
                    <span>Max Output</span>
                  </div>
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <p className="text-sm font-medium text-white cursor-help">
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
                  <div className="flex items-center gap-1 text-xs text-gray-400">
                    <DollarSign className="h-3 w-3" />
                    <span>Cost</span>
                  </div>
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <p className="text-sm font-medium text-white cursor-help">
                          ${selectedModel.input_cost_per_1k.toFixed(4)}/1K
                        </p>
                      </TooltipTrigger>
                      <TooltipContent>
                        <div className="text-xs space-y-1">
                          <div>Input: ${selectedModel.input_cost_per_1k.toFixed(4)}/1K tokens</div>
                          <div>Output: ${selectedModel.output_cost_per_1k.toFixed(4)}/1K tokens</div>
                        </div>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>
              </div>

              {/* Capabilities */}
              {Object.keys(selectedModel.capabilities).length > 0 && (
                <div className="space-y-2">
                  <div className="flex items-center gap-1 text-xs text-gray-400">
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
                  <Badge variant="outline" className="border-blue-500/30 text-blue-400 text-xs">
                    <Check className="h-3 w-3 mr-1" />
                    Function Calling
                  </Badge>
                )}
                {selectedModel.supports_vision && (
                  <Badge variant="outline" className="border-purple-500/30 text-purple-400 text-xs">
                    <Check className="h-3 w-3 mr-1" />
                    Vision
                  </Badge>
                )}
                {selectedModel.supports_streaming && (
                  <Badge variant="outline" className="border-gray-500/30 text-gray-400 text-xs">
                    <Check className="h-3 w-3 mr-1" />
                    Streaming
                  </Badge>
                )}
              </div>

              {/* Recommended For */}
              {selectedModel.recommended_for.length > 0 && (
                <div className="space-y-2">
                  <div className="text-xs text-gray-400">Recommended for:</div>
                  <div className="flex flex-wrap gap-1">
                    {selectedModel.recommended_for.map((task) => (
                      <Badge 
                        key={task} 
                        variant="secondary"
                        className="text-xs bg-gray-800/50 text-gray-300"
                      >
                        {task.replace(/_/g, ' ')}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}

              {/* Description */}
              {selectedModel.description && (
                <p className="text-xs text-gray-400 leading-relaxed">
                  {selectedModel.description}
                </p>
              )}

              {/* Warning for deprecated */}
              {selectedModel.status === 'deprecated' && (
                <div className="flex items-start gap-2 p-2 rounded-lg bg-yellow-500/10 border border-yellow-500/20">
                  <AlertCircle className="h-4 w-4 text-yellow-400 mt-0.5 flex-shrink-0" />
                  <div className="text-xs text-yellow-400">
                    This model is deprecated and may be removed in the future.
                  </div>
                </div>
              )}
              
              {/* Beta warning */}
              {selectedModel.status === 'beta' && (
                <div className="flex items-start gap-2 p-2 rounded-lg bg-blue-500/10 border border-blue-500/20">
                  <Info className="h-4 w-4 text-blue-400 mt-0.5 flex-shrink-0" />
                  <div className="text-xs text-blue-400">
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

