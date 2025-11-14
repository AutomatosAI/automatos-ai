'use client'

import { useState, useRef, useCallback, useEffect } from 'react'
import { Send, StopCircle, Paperclip } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { ModelSelector } from './model-selector'
import type { VisibilityType, AppUsage } from '@/types'

export interface MultimodalInputProps {
  chatId: string
  status: any
  stop: () => void
  sendMessage: (message: any) => void
  selectedModelId: string
  onModelChange: (modelId: string) => void
  selectedVisibilityType: VisibilityType
  usage?: AppUsage
}

export function MultimodalInput({
  chatId,
  status,
  stop,
  sendMessage,
  selectedModelId,
  onModelChange,
  selectedVisibilityType,
  usage,
}: MultimodalInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const [attachments, setAttachments] = useState<File[]>([])
  const [input, setInput] = useState('')
  
  // Safe input with default
  const safeInput = input || ''

  const adjustHeight = useCallback(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`
    }
  }, [])

  useEffect(() => {
    adjustHeight()
  }, [safeInput, adjustHeight])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    const trimmedInput = safeInput.trim()
    if (!trimmedInput || status === 'streaming') return

    // Send message as proper object
    sendMessage({
      role: 'user',
      content: trimmedInput,
    })
    
    // Clear input and attachments
    setInput('')
    setAttachments([])
    
    // Reset height
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  const isStreaming = status === 'streaming'

  return (
    <form onSubmit={handleSubmit} className="w-full">
      {/* Large input box with everything inside - ai-chatbot style */}
      <div className="relative w-full rounded-3xl border border-orange-500/10 bg-transparent focus-within:border-orange-500/30 transition-all shadow-sm">
        {/* Textarea */}
        <Textarea
          ref={textareaRef}
          value={safeInput}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Send a message..."
          className="min-h-[100px] max-h-[200px] w-full resize-none bg-transparent border-0 px-4 pt-4 pb-14 text-base text-foreground placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-0"
          rows={1}
        />
        
        {/* Bottom toolbar - inside the input */}
        <div className="absolute bottom-0 left-0 right-0 flex items-center justify-between px-3 py-2 border-t border-transparent">
          {/* Left side: Attachment + Model Selector */}
          <div className="flex items-center gap-2">
            <Button
              type="button"
              variant="ghost"
              size="sm"
              className="h-8 w-8 p-0 text-muted-foreground hover:text-foreground"
              disabled={isStreaming}
            >
              <Paperclip className="w-4 h-4" />
            </Button>
            
            <ModelSelector
              selectedModelId={selectedModelId}
              onModelChange={onModelChange}
            />
          </div>
          
          {/* Right side: Send/Stop Button */}
          {isStreaming ? (
            <Button
              type="button"
              onClick={stop}
              size="icon"
              className="bg-red-600 hover:bg-red-700 h-8 w-8 rounded-lg"
            >
              <StopCircle className="w-4 h-4" />
            </Button>
          ) : (
            <Button
              type="submit"
              disabled={!safeInput.trim()}
              size="icon"
              className="bg-gradient-to-br from-orange-500 to-red-500 hover:from-orange-600 hover:to-red-600 h-8 w-8 rounded-lg disabled:opacity-50"
            >
              <Send className="w-4 h-4" />
            </Button>
          )}
        </div>
      </div>

      {/* Usage Info */}
      {usage && usage.totalTokens && (
        <div className="flex items-center justify-between text-xs text-gray-500">
          <span>Tokens: {usage.totalTokens.toLocaleString()}</span>
          {usage.cost && <span>Cost: ${usage.cost.toFixed(4)}</span>}
        </div>
      )}
    </form>
  )
}

