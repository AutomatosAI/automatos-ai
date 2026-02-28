'use client'

import { Check, ChevronDown, Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
  DropdownMenuSeparator,
  DropdownMenuLabel,
} from '@/components/ui/dropdown-menu'
import { useWorkspaceModels } from '@/hooks/use-model-api'
import { useWorkspace } from '@/components/workspace-provider'
import { chatModels, getModelById } from '@/lib/ai/models'

export interface ModelSelectorProps {
  selectedModelId: string
  onModelChange: (modelId: string) => void
}

export function ModelSelector({ selectedModelId, onModelChange }: ModelSelectorProps) {
  // Load workspace-installed models (same catalog agents use)
  const { isLoading: isWorkspaceLoading } = useWorkspace()
  const { data: apiModels = [], isLoading } = useWorkspaceModels({ enabled: !isWorkspaceLoading })

  // Use API models if available, fallback to hardcoded for offline
  const models = apiModels.length > 0 ? apiModels : chatModels

  // Find selected model (check both API format and local format)
  const selectedModel = models.find((m: any) =>
    m.model_id === selectedModelId || m.id === selectedModelId
  ) || getModelById(selectedModelId)

  // Group models by provider
  const modelsByProvider = models.reduce((acc: Record<string, any[]>, model: any) => {
    const provider = model.provider || 'other'
    if (!acc[provider]) acc[provider] = []
    acc[provider].push(model)
    return acc
  }, {})

  const providerLabels: Record<string, string> = {
    openai: '🤖 OpenAI',
    anthropic: '🧠 Anthropic',
    grok: '⚡ xAI (Grok)',
    huggingface: '🤗 HuggingFace',
    google: '🔮 Google',
    other: '📦 Other'
  }

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant="ghost"
          size="sm"
          className="h-8 px-3 rounded-full border-2 border-orange-500/20 bg-black/20 hover:bg-orange-500/5 hover:border-orange-500/40 text-foreground/90 text-xs gap-2 shadow-[0_0_18px_rgba(249,115,22,0.10)]"
        >
          {isLoading ? (
            <Loader2 className="w-3 h-3 animate-spin" />
          ) : (
            <span className="truncate max-w-[160px]">
              {selectedModel?.display_name || selectedModel?.name || 'Select Model'}
            </span>
          )}
          <ChevronDown className="w-3 h-3 text-orange-400" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent
        align="start"
        className="w-[340px] max-h-[500px] overflow-y-auto bg-background border-orange-500/20 rounded-2xl"
        style={{ maxHeight: '70vh' }}
      >
        {Object.entries(modelsByProvider).map(([provider, providerModels]) => (
          <div key={provider}>
            <DropdownMenuLabel className="text-xs text-muted-foreground sticky top-0 bg-background z-10">
              {providerLabels[provider] || provider}
            </DropdownMenuLabel>
            {(providerModels as any[]).map((model: any) => {
              const modelId = model.model_id || model.id
              const modelName = model.display_name || model.name
              const modelDesc = model.description || ''
              const isSelected = selectedModelId === modelId

              return (
                <DropdownMenuItem
                  key={modelId}
                  onClick={() => onModelChange(modelId)}
                  className="text-foreground hover:bg-orange-500/10 cursor-pointer rounded-lg py-2"
                >
                  <div className="flex items-start justify-between w-full">
                    <div className="flex-1 min-w-0">
                      <div className="font-medium text-sm">{modelName}</div>
                      {modelDesc && (
                        <div className="text-xs text-muted-foreground truncate">
                          {modelDesc}
                        </div>
                      )}
                    </div>
                    {isSelected && (
                      <Check className="w-4 h-4 text-orange-400 ml-2 flex-shrink-0" />
                    )}
                  </div>
                </DropdownMenuItem>
              )
            })}
            <DropdownMenuSeparator />
          </div>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
