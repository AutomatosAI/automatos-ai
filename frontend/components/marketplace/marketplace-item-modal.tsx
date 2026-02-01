'use client'

import { useState, useEffect } from 'react'
import { X, Download, Zap, Wrench, Brain, Settings, Info } from 'lucide-react'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import { useToast } from '@/components/ui/use-toast'
import type { MarketplaceItem } from './marketplace-homepage'
import { apiClient } from '@/lib/api-client'
import { toast as sonnerToast } from 'sonner'
import { ToolLogo } from '@/components/ui/tool-logo'
import { useAvailableApps } from '@/hooks/use-composio-api'

interface MarketplaceItemModalProps {
  itemId: number
  onClose: () => void
}

export function MarketplaceItemModal({
  itemId,
  onClose
}: MarketplaceItemModalProps) {
  const [item, setItem] = useState<MarketplaceItem | null>(null)
  const [loading, setLoading] = useState(true)
  const [installing, setInstalling] = useState(false)
  const { toast } = useToast()

  // Fetch available Composio apps for tool descriptions
  const { data: availableApps = [] } = useAvailableApps()

  useEffect(() => {
    fetchItem()
  }, [itemId])

  async function fetchItem() {
    try {
      const data = await apiClient.get(`/api/marketplace/items/${itemId}`)
      setItem(data)
    } catch (error) {
      console.error('Failed to fetch item details:', error)
      toast({
        title: 'Error',
        description: 'Failed to load item details',
        variant: 'destructive'
      })
    } finally {
      setLoading(false)
    }
  }

  async function handleInstall() {
    if (!item) return

    setInstalling(true)
    try {
      const result = await apiClient.post(`/api/marketplace/items/${item.id}/install`)
      sonnerToast.success('Agent added to workspace!', {
        description: result.message || `${item.name} has been added to your workspace.`
      })
      onClose()
    } catch (error: any) {
      console.error('Failed to install item:', error)
      sonnerToast.error('Failed to add to workspace', {
        description: error?.message || 'Failed to add agent to workspace. Please try again.'
      })
    } finally {
      setInstalling(false)
    }
  }

  // Get skills from metadata (full details now included by backend)
  const skills = item?.metadata?.skills || item?.dependencies?.skills || []

  // Get tools from metadata (now includes logos and descriptions from backend)
  const toolNames = item?.metadata?.tool_names || []
  const toolIcons = item?.metadata?.tool_icons || []
  const toolDescriptions = item?.metadata?.tool_descriptions || []

  // Get configuration details
  const modelConfig = item?.metadata?.model_config || {}
  const llmConfig = item?.metadata?.configuration?.llm_config || {}
  const capabilities = item?.metadata?.configuration?.capabilities || []

  // Get model info (prefer model_config, fallback to llm_config)
  const modelProvider = modelConfig.provider || llmConfig.provider || 'openai'
  const modelId = modelConfig.model_id || llmConfig.model || 'gpt-4'
  const temperature = modelConfig.temperature || llmConfig.temperature
  const maxTokens = modelConfig.max_tokens || llmConfig.max_tokens
  const contextWindow = llmConfig.context_window

  if (loading || !item) {
    return (
      <Dialog open={true} onOpenChange={onClose}>
        <DialogContent className="glass-card border-border/50">
          <div className="animate-pulse space-y-4">
            <div className="h-8 bg-secondary/50 rounded" />
            <div className="h-32 bg-secondary/50 rounded" />
            <div className="h-24 bg-secondary/50 rounded" />
          </div>
        </DialogContent>
      </Dialog>
    )
  }

  return (
    <Dialog open={true} onOpenChange={onClose}>
      <DialogContent className="max-w-5xl max-h-[90vh] overflow-hidden glass-card border-border/50">
        <DialogHeader className="border-b border-border/30 pb-4">
          <div className="flex items-start justify-between gap-4">
            <div className="flex items-center gap-4 flex-1 min-w-0">
              {item.icon && <div className="text-5xl flex-shrink-0">{item.icon}</div>}
              <div className="flex-1 min-w-0">
                <DialogTitle className="text-2xl font-bold">{item.name}</DialogTitle>
                <div className="flex items-center gap-3 mt-2 text-sm text-muted-foreground">
                  <span className="truncate">by {item.creator_name}</span>
                  <span>•</span>
                  <Badge variant="outline" className="text-xs border-orange-500/30 text-orange-400">
                    {item.category}
                  </Badge>
                  <span>•</span>
                  <div className="flex items-center gap-1">
                    <Download className="w-3 h-3" />
                    <span>{item.install_count} installs</span>
                  </div>
                </div>
              </div>
            </div>
            <div className="flex items-center gap-2 flex-shrink-0">
              <Button
                onClick={handleInstall}
                disabled={installing}
                className="bg-gradient-to-r from-orange-500 to-red-500 hover:from-orange-600 hover:to-red-600 text-white"
              >
                <Download className="w-4 h-4 mr-2" />
                {installing ? 'Adding...' : 'Add to Workspace'}
              </Button>
              <Button
                variant="ghost"
                size="icon"
                onClick={onClose}
                className="h-8 w-8"
              >
                <X className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </DialogHeader>

        <div className="overflow-y-auto max-h-[calc(90vh-180px)] p-6 space-y-6">
          {/* Description */}
          <div>
            <p className="text-muted-foreground leading-relaxed">
              {item.description || 'No description available'}
            </p>
          </div>

          <Separator />

          {/* LLM Model Configuration */}
          <div className="bg-secondary/30 border border-border/30 rounded-lg p-4">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xl font-bold uppercase">{modelId}</h3>
              <Badge variant="outline" className="capitalize">
                {modelProvider}
              </Badge>
            </div>

            <p className="text-sm text-muted-foreground mb-4">
              {modelId.toLowerCase().includes('gpt-4')
                ? 'Most capable GPT-4 model for complex reasoning and coding tasks'
                : modelId.toLowerCase().includes('claude')
                ? 'Advanced Claude model with strong reasoning capabilities'
                : 'AI model optimized for agent workflows'}
            </p>

            <div className="grid grid-cols-3 gap-4 mb-4">
              {contextWindow && (
                <div>
                  <div className="flex items-center gap-1 text-sm text-muted-foreground mb-1">
                    <Info className="w-3 h-3" />
                    <span>Context</span>
                  </div>
                  <p className="font-semibold">{contextWindow >= 1000 ? `${contextWindow / 1000}K` : contextWindow} tokens</p>
                </div>
              )}
              {maxTokens && (
                <div>
                  <div className="flex items-center gap-1 text-sm text-muted-foreground mb-1">
                    <Zap className="w-3 h-3" />
                    <span>Max Output</span>
                  </div>
                  <p className="font-semibold">{maxTokens >= 1000 ? `${maxTokens / 1000}K` : maxTokens} tokens</p>
                </div>
              )}
              {temperature !== undefined && (
                <div>
                  <div className="flex items-center gap-1 text-sm text-muted-foreground mb-1">
                    <Settings className="w-3 h-3" />
                    <span>Temperature</span>
                  </div>
                  <p className="font-semibold">{temperature}</p>
                </div>
              )}
            </div>

            {capabilities.length > 0 && (
              <>
                <div className="flex items-center gap-1 text-sm text-muted-foreground mb-2">
                  <Info className="w-3 h-3" />
                  <span>Capabilities</span>
                </div>
                <div className="flex flex-wrap gap-2 mb-4">
                  {capabilities.map((cap: string, i: number) => (
                    <Badge key={i} variant="outline" className="border-green-500/30 text-green-400">
                      {cap.replace(/_/g, ' ')}: excellent
                    </Badge>
                  ))}
                </div>
              </>
            )}

            {(modelConfig.top_p || modelConfig.frequency_penalty !== undefined) && (
              <div className="flex flex-wrap gap-4 text-sm">
                {modelConfig.top_p && (
                  <div className="flex items-center gap-2">
                    <span className="text-blue-400">✓</span>
                    <span>Function Calling</span>
                  </div>
                )}
                <div className="flex items-center gap-2">
                  <span className="text-blue-400">✓</span>
                  <span>Streaming</span>
                </div>
              </div>
            )}
          </div>

          {/* Assigned Skills */}
          {skills && skills.length > 0 && (
            <>
              <Separator />
              <div>
                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                  <Brain className="w-5 h-5" />
                  Assigned Skills
                </h3>
                <div className="space-y-3">
                  {skills.map((skill: any, idx: number) => (
                    <div key={idx} className="bg-secondary/30 border border-border/30 rounded-lg p-4">
                      <div className="flex items-start justify-between mb-2">
                        <h4 className="font-semibold">{skill.name || `Skill ${idx + 1}`}</h4>
                        {skill.category && (
                          <Badge variant="outline" className="text-xs">
                            {skill.category}
                          </Badge>
                        )}
                      </div>
                      {skill.description && (
                        <p className="text-sm text-muted-foreground leading-relaxed">
                          {skill.description}
                        </p>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}

          {/* Assigned Tools */}
          {toolNames && toolNames.length > 0 && (
            <>
              <Separator />
              <div>
                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                  <Wrench className="w-5 h-5" />
                  Assigned Tools
                </h3>
                <div className="space-y-3">
                  {toolNames.map((toolName: string, idx: number) => {
                    // Use data from backend (logos and descriptions now included)
                    const logo = toolIcons?.[idx]
                    const description = toolDescriptions?.[idx]

                    return (
                      <div
                        key={idx}
                        className="bg-secondary/30 border border-border/30 rounded-lg p-4"
                      >
                        <div className="flex items-start justify-between mb-2">
                          <div className="flex items-center gap-3">
                            <ToolLogo
                              name={toolName}
                              logo={logo}
                              size={24}
                            />
                            <h4 className="font-semibold uppercase">
                              {toolName}
                            </h4>
                          </div>
                          <Badge variant="outline" className="text-xs">
                            Composio
                          </Badge>
                        </div>
                        {description && (
                          <p className="text-sm text-muted-foreground leading-relaxed">
                            {description}
                          </p>
                        )}
                      </div>
                    )
                  })}
                </div>
              </div>
            </>
          )}
        </div>
      </DialogContent>
    </Dialog>
  )
}
