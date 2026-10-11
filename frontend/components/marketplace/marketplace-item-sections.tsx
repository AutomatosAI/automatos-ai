'use client'

import { useEffect, useState } from 'react'
import { LLM_DEFAULTS } from '@/lib/llm-defaults'
import { Download, Zap, Wrench, Brain, Settings, Info } from 'lucide-react'
import { DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import type { MarketplaceItem } from './marketplace-homepage'
import { apiClient } from '@/lib/api-client'
import { toast as sonnerToast } from 'sonner'
import { ToolLogo } from '@/components/ui/tool-logo'
import { useInstallPlaybookFromMarketplace } from '@/hooks/use-playbook-api'

/** 'recipe' is the marketplace API's value for a playbook. */
export const isPlaybookType = (itemType: string) => itemType === 'recipe'

/** Loads one marketplace item by its type and id (agents and playbooks share id values). */
export function useMarketplaceItemDetail(itemId: number, itemType: string) {
  const [item, setItem] = useState<MarketplaceItem | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    setError(null)
    apiClient.get(`/api/marketplace/items/${itemId}?type=${encodeURIComponent(itemType)}`)
      .then((data: MarketplaceItem) => { if (!cancelled) setItem(data) })
      .catch((err: any) => {
        if (cancelled) return
        const message = err?.message || 'Failed to load item details'
        setError(message)
        sonnerToast.error('Error', { description: message })
      })
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [itemId, itemType])

  return { item, loading, error }
}

/** Installs the item through its own route: playbooks and agents install differently. */
export function useInstallMarketplaceDetailItem(itemType: string, onDone: () => void) {
  const [installing, setInstalling] = useState(false)
  const installPlaybook = useInstallPlaybookFromMarketplace()
  const noun = isPlaybookType(itemType) ? 'Playbook' : 'Agent'

  async function install(item: MarketplaceItem) {
    setInstalling(true)
    try {
      const result: any = isPlaybookType(itemType)
        ? await installPlaybook.mutateAsync(item.id)
        : await apiClient.post(`/api/marketplace/items/${item.id}/install`)
      sonnerToast.success(`${noun} added to workspace!`, {
        description: result?.message || `${item.name} has been added to your workspace.`
      })
      onDone()
    } catch (error: any) {
      sonnerToast.error('Failed to add to workspace', {
        description: error?.message || `Failed to add ${noun.toLowerCase()} to workspace. Please try again.`
      })
    } finally {
      setInstalling(false)
    }
  }

  return { installing, install }
}

export function ItemModalHeader({ item, installing, onInstall }: {
  item: MarketplaceItem
  installing: boolean
  onInstall: () => void
}) {
  return (
    <DialogHeader className="border-b border-border/30 pb-4">
      <div className="flex items-start justify-between gap-4">
        <div className="flex items-center gap-4 flex-1 min-w-0">
          {item.icon && <div className="text-5xl flex-shrink-0">{item.icon}</div>}
          <div className="flex-1 min-w-0">
            <DialogTitle className="text-2xl font-bold">{item.name}</DialogTitle>
            <div className="flex items-center gap-3 mt-2 text-sm text-muted-foreground">
              <span className="truncate">by {item.creator_name}</span>
              <span>•</span>
              <Badge variant="outline" className="text-xs border-primary/30 text-primary">
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
          <Button onClick={onInstall} disabled={installing}>
            <Download className="w-4 h-4 mr-2" />
            {installing ? 'Adding...' : 'Add to Workspace'}
          </Button>
        </div>
      </div>
    </DialogHeader>
  )
}

const formatTokens = (n: number) => (n >= 1000 ? `${n / 1000}K` : n)

function modelSummary(modelId: string) {
  const id = modelId.toLowerCase()
  if (id.includes('gpt-4')) return 'Most capable GPT-4 model for complex reasoning and coding tasks'
  if (id.includes('claude')) return 'Advanced Claude model with strong reasoning capabilities'
  return 'AI model optimized for agent workflows'
}

function ModelStat({ icon: Icon, label, value }: { icon: typeof Info; label: string; value: React.ReactNode }) {
  return (
    <div>
      <div className="flex items-center gap-1 text-sm text-muted-foreground mb-1">
        <Icon className="w-3 h-3" />
        <span>{label}</span>
      </div>
      <p className="font-semibold">{value}</p>
    </div>
  )
}

export function ItemModelSection({ item }: { item: MarketplaceItem }) {
  // Prefer model_config, fall back to the agent's llm_config.
  const modelConfig = item.metadata?.model_config || {}
  const llmConfig = item.metadata?.configuration?.llm_config || {}
  const capabilities: string[] = item.metadata?.configuration?.capabilities || []
  const modelProvider = modelConfig.provider || llmConfig.provider || 'openrouter'
  const modelId: string = modelConfig.model_id || llmConfig.model || LLM_DEFAULTS.model_id
  const temperature = modelConfig.temperature || llmConfig.temperature
  const maxTokens = modelConfig.max_tokens || llmConfig.max_tokens
  const contextWindow = llmConfig.context_window

  return (
    <div className="bg-secondary/30 border border-border/30 rounded-lg p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-xl font-bold uppercase break-all">{modelId}</h3>
        <Badge variant="outline" className="capitalize">{modelProvider}</Badge>
      </div>

      <p className="text-sm text-muted-foreground mb-4">{modelSummary(modelId)}</p>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-4">
        {contextWindow && <ModelStat icon={Info} label="Context" value={`${formatTokens(contextWindow)} tokens`} />}
        {maxTokens && <ModelStat icon={Zap} label="Max Output" value={`${formatTokens(maxTokens)} tokens`} />}
        {temperature !== undefined && <ModelStat icon={Settings} label="Temperature" value={temperature} />}
      </div>

      {capabilities.length > 0 && (
        <>
          <div className="flex items-center gap-1 text-sm text-muted-foreground mb-2">
            <Info className="w-3 h-3" />
            <span>Capabilities</span>
          </div>
          <div className="flex flex-wrap gap-2 mb-4">
            {capabilities.map((cap, i) => (
              <Badge key={i} variant="outline" className="border-[hsl(var(--success))]/30 text-[hsl(var(--success))]">
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
              <span className="text-[hsl(var(--info))]">✓</span>
              <span>Function Calling</span>
            </div>
          )}
          <div className="flex items-center gap-2">
            <span className="text-[hsl(var(--info))]">✓</span>
            <span>Streaming</span>
          </div>
        </div>
      )}
    </div>
  )
}

export function ItemSkillsSection({ skills }: { skills: any[] }) {
  if (!skills || skills.length === 0) return null
  return (
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
                {skill.category && <Badge variant="outline" className="text-xs">{skill.category}</Badge>}
              </div>
              {skill.description && (
                <p className="text-sm text-muted-foreground leading-relaxed">{skill.description}</p>
              )}
            </div>
          ))}
        </div>
      </div>
    </>
  )
}

export function ItemToolsSection({ item }: { item: MarketplaceItem }) {
  // Logos and descriptions come from the backend alongside the names.
  const toolNames: string[] = item.metadata?.tool_names || []
  const toolIcons: string[] = item.metadata?.tool_icons || []
  const toolDescriptions: string[] = item.metadata?.tool_descriptions || []
  if (toolNames.length === 0) return null

  return (
    <>
      <Separator />
      <div>
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Wrench className="w-5 h-5" />
          Assigned Tools
        </h3>
        <div className="space-y-3">
          {toolNames.map((toolName, idx) => (
            <div key={idx} className="bg-secondary/30 border border-border/30 rounded-lg p-4">
              <div className="flex items-start justify-between mb-2">
                <div className="flex items-center gap-3">
                  <ToolLogo name={toolName} logo={toolIcons[idx]} size={24} />
                  <h4 className="font-semibold uppercase">{toolName}</h4>
                </div>
                <Badge variant="outline" className="text-xs">Composio</Badge>
              </div>
              {toolDescriptions[idx] && (
                <p className="text-sm text-muted-foreground leading-relaxed">{toolDescriptions[idx]}</p>
              )}
            </div>
          ))}
        </div>
      </div>
    </>
  )
}
