'use client'

import { useState, useRef, useCallback, useEffect } from 'react'
import { Send, StopCircle, Paperclip } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { ModelSelector } from './model-selector'
import { AgentSelector, type Agent } from './agent-selector'
import { ToolLogo } from '@/components/ui/tool-logo'
import type { VisibilityType, AppUsage } from '@/types'
import { apiClient } from '@/lib/api-client'
import { toast } from 'sonner'

export interface MultimodalInputProps {
  chatId: string
  status: any
  stop: () => void
  sendMessage: (message: any) => void
  selectedModelId: string
  onModelChange: (modelId: string) => void
  selectedAgentId?: number | null
  onAgentChange?: (agentId: number | null) => void
  selectedVisibilityType: VisibilityType
  usage?: AppUsage
  // PRD-40: Tool icon click handler
  onToolIconClick?: (appName: string) => void
}

export function MultimodalInput({
  chatId,
  status,
  stop,
  sendMessage,
  selectedModelId,
  onModelChange,
  selectedAgentId,
  onAgentChange,
  selectedVisibilityType,
  usage,
  onToolIconClick,
}: MultimodalInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [uploadQueue, setUploadQueue] = useState<string[]>([])
  const [uploadedDocs, setUploadedDocs] = useState<Array<{ document_id: string; filename: string; status: string }>>([])
  const [input, setInput] = useState('')
  const [activeAgent, setActiveAgent] = useState<Agent | null>(null)

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

  // Refocus textarea when streaming finishes
  useEffect(() => {
    if (status !== 'streaming') {
      textareaRef.current?.focus()
    }
  }, [status])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    const trimmedInput = safeInput.trim()
    if (!trimmedInput || status === 'streaming') return

    const parts: any[] = [
      ...uploadedDocs.map((doc) => ({
        type: 'file',
        filename: doc.filename,
        mediaType: 'application/octet-stream',
        url: `document://${doc.document_id}`,
      })),
      { type: 'text', text: trimmedInput },
    ]

    // Send message with parts (text + uploaded document references)
    sendMessage({
      role: 'user',
      content: trimmedInput,
      parts,
    })

    // Clear input and attachments
    setInput('')
    setUploadedDocs([])

    // Reset height and refocus
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.focus()
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
    <form onSubmit={handleSubmit} className="w-full" data-tour="chat-input-area">
      <input
        ref={fileInputRef}
        type="file"
        className="hidden"
        multiple
        onChange={async (event) => {
          const files = Array.from(event.target.files || [])
          if (files.length === 0) return

          setUploadQueue(files.map((f) => f.name))

          try {
            const results = await Promise.all(
              files.map((file) => apiClient.uploadDocument(file))
            )

            setUploadedDocs((prev) => [
              ...prev,
              ...results.map((r: any, idx: number) => ({
                document_id: String(r.document_id ?? r.id ?? ''),
                filename: String(r.filename ?? files[idx]?.name ?? 'document'),
                status: String(r.status ?? 'uploaded'),
              })),
            ])

            toast.success(`Uploaded ${files.length} document${files.length === 1 ? '' : 's'}`)
          } catch (error: any) {
            console.error('Document upload failed', error)
            toast.error(error?.message || 'Failed to upload document(s)')
          } finally {
            setUploadQueue([])
            // Reset file input so selecting the same file again works
            if (fileInputRef.current) fileInputRef.current.value = ''
          }
        }}
      />

      {(uploadedDocs.length > 0 || uploadQueue.length > 0) && (
        <div className="mb-3 flex flex-wrap gap-2">
          {uploadedDocs.map((doc) => (
            <div
              key={`${doc.document_id}-${doc.filename}`}
              className="inline-flex items-center gap-2 rounded-full border border-blue-500/20 bg-blue-500/10 px-3 py-1 text-xs text-blue-200"
            >
              <span className="truncate max-w-[240px]">{doc.filename}</span>
              <span className="text-blue-100/70">({doc.status})</span>
              <button
                type="button"
                className="ml-1 text-blue-100/70 hover:text-blue-100"
                onClick={() => setUploadedDocs((prev) => prev.filter((d) => d.document_id !== doc.document_id))}
                aria-label="Remove attachment"
              >
                ×
              </button>
            </div>
          ))}

          {uploadQueue.map((name) => (
            <div
              key={name}
              className="inline-flex items-center gap-2 rounded-full border border-orange-500/20 bg-orange-500/10 px-3 py-1 text-xs text-orange-200"
            >
              <span className="truncate max-w-[240px]">{name}</span>
              <span className="text-orange-100/70">(uploading…)</span>
            </div>
          ))}
        </div>
      )}

      {/* Large input box with everything inside - Incredible-style centered card */}
      <div
        className={[
          'relative w-full rounded-3xl border-2 border-orange-500/20 bg-transparent',
          'focus-within:border-orange-500/40 focus-within:ring-2 focus-within:ring-orange-500/15',
          'transition-all shadow-[0_0_60px_rgba(249,115,22,0.08)]',
        ].join(' ')}
      >
        {/* Textarea */}
        <Textarea
          ref={textareaRef}
          value={safeInput}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Send a message..."
          className="min-h-[60px] max-h-[200px] w-full resize-none rounded-3xl bg-transparent border-0 px-4 pt-4 pb-14 text-base text-foreground placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-0 focus-visible:ring-offset-0"
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
              disabled={isStreaming || uploadQueue.length > 0}
              onClick={() => fileInputRef.current?.click()}
            >
              <Paperclip className="w-4 h-4" />
            </Button>

            {/* PRD: Unified Agent-Chat System - Agent Selector */}
            {onAgentChange ? (
              <AgentSelector
                selectedAgentId={selectedAgentId}
                onAgentChange={onAgentChange}
                onAgentData={setActiveAgent}
              />
            ) : (
              <ModelSelector
                selectedModelId={selectedModelId}
                onModelChange={onModelChange}
              />
            )}
          </div>

          {/* Active Agent Tools - Next to Send Button */}
          <div className="flex items-center gap-3 ml-auto mr-2 mt-1">
            {activeAgent?.tools && activeAgent.tools.length > 0 && (
              <div className="flex gap-2 items-center animate-in fade-in zoom-in-50 duration-300">
                {activeAgent.tools.slice(0, 4).map((tool) => (
                  <button
                    key={tool.id}
                    type="button"
                    onClick={() => onToolIconClick?.(tool.name)}
                    className="relative hover:scale-110 transition-all duration-200 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 rounded-lg"
                    title={`Click for ${tool.name} suggestions`}
                    aria-label={`Show suggestions for ${tool.name}`}
                  >
                    <ToolLogo
                      name={tool.name}
                      logo={tool.icon}
                      size={28}
                      showBackground={true}
                      className="bg-background/80 ring-1 ring-orange-500/30 border border-orange-500/20 shadow-[0_0_10px_rgba(249,115,22,0.1)] rounded-lg"
                    />
                  </button>
                ))}
                {activeAgent.tools.length > 4 && (
                  <div className="w-7 h-7 rounded-lg bg-secondary/80 ring-1 ring-orange-500/30 flex items-center justify-center text-[10px] font-bold text-muted-foreground border border-orange-500/20 relative z-10">
                    +{activeAgent.tools.length - 4}
                  </div>
                )}
              </div>
            )}

            {/* Send/Stop Button */}
            {isStreaming ? (
              <Button
                type="button"
                onClick={stop}
                size="icon"
                className="bg-red-600 hover:bg-red-700 h-9 w-9 rounded-xl shadow-sm"
              >
                <StopCircle className="w-4 h-4" />
              </Button>
            ) : (
              <Button
                type="submit"
                disabled={!safeInput.trim()}
                size="icon"
                className="h-10 w-10 rounded-2xl disabled:opacity-50 bg-gradient-to-br from-orange-500 to-red-500 hover:from-orange-600 hover:to-red-600 shadow-[0_0_20px_rgba(249,115,22,0.25)] hover:shadow-[0_0_28px_rgba(249,115,22,0.35)] transition-shadow"
              >
                <Send className="w-4 h-4" />
              </Button>
            )}
          </div>
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

