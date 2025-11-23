'use client'

import { Check, ChevronDown } from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { chatModels, getModelById } from '@/lib/ai/models'

export interface ModelSelectorProps {
  selectedModelId: string
  onModelChange: (modelId: string) => void
}

export function ModelSelector({ selectedModelId, onModelChange }: ModelSelectorProps) {
  const selectedModel = getModelById(selectedModelId)

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant="ghost"
          size="sm"
          className="h-8 px-2 rounded-lg border border-orange-500/20 hover:border-orange-500/40 hover:bg-orange-500/5 text-muted-foreground text-xs gap-1"
        >
          <span className="truncate">{selectedModel?.name || 'GPT-4'}</span>
          <ChevronDown className="w-3 h-3" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="w-[240px] bg-background border-orange-500/20 rounded-xl">
        {chatModels.map((model) => (
          <DropdownMenuItem
            key={model.id}
            onClick={() => onModelChange(model.id)}
            className="text-foreground hover:bg-orange-500/10 cursor-pointer rounded-lg"
          >
            <div className="flex items-start justify-between w-full">
              <div className="flex-1">
                <div className="font-medium">{model.name}</div>
                <div className="text-xs text-gray-500">{model.description}</div>
              </div>
              {selectedModelId === model.id && (
                <Check className="w-4 h-4 text-orange-400 ml-2" />
              )}
            </div>
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

