'use client'

import { useState, useRef, useCallback, useEffect } from 'react'
import { Send, StopCircle, Paperclip } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
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
  setMessages?: React.Dispatch<React.SetStateAction<any[]>>
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
  setMessages,
  selectedAgentId,
  onAgentChange,
  selectedVisibilityType,
  usage,
  onToolIconClick,
}: MultimodalInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [uploadQueue, setUploadQueue] = useState<string[]>([])
  // PRD-127: Switch to ephemeral attachments
  const [uploadedAttachments, setUploadedAttachments] = useState<Array<{
    attachment_id: string
    filename: string
    media_type: 'image' | 'document'
  }>>([])
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

    // PRD-127: Send attachment_ids instead of document:// URLs
    const attachment_ids = uploadedAttachments.map((att) => att.attachment_id)

    // Build message payload with attachment_ids
    sendMessage({
      role: 'user',
      content: trimmedInput,
      attachment_ids,  // PRD-127: new field for ephemeral attachments
      // Keep parts for display purposes (filename chips in message history)
      parts: [
        ...uploadedAttachments.map((att) => ({
          type: 'file',
          filename: att.filename,
          mediaType: att.media_type === 'image' ? 'image/*' : 'application/octet-stream',
          attachment_id: att.attachment_id,
        })),
        { type: 'text', text: trimmedInput },
      ],
    })

    // Clear input and attachments
    setInput('')
    setUploadedAttachments([])

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
      {/* PRD-127: Ephemeral attachment upload */}
      <input
        ref={fileInputRef}
        type="file"
        className="hidden"
        multiple
        accept="image/*,.pdf,.doc,.docx,.xls,.xlsx,.txt,.csv,.md,.json,.py,.js,.ts,.tsx"
        onChange={async (event) => {
          const files = Array.from(event.target.files || [])
          if (files.length === 0) return

          setUploadQueue(files.map((f) => f.name))

          try {
            // PRD-127: Use uploadAttachment instead of uploadDocument
            const results = await Promise.all(
              files.map((file) => apiClient.uploadAttachment(file))
            )

            setUploadedAttachments((prev) => [
              ...prev,
              ...results.map((r) => ({
                attachment_id: r.attachment_id,
                filename: r.filename,
                media_type: r.media_type,
              })),
            ])

            toast.success(`Uploaded ${files.length} file${files.length === 1 ? '' : 's'}`)
          } catch (error: unknown) {
            const msg = error instanceof Error ? error.message : 'Upload failed'
            toast.error(msg)
          } finally {
            setUploadQueue([])
            // Reset file input so selecting the same file again works
            if (fileInputRef.current) fileInputRef.current.value = ''
          }
        }}
      />

      {/* PRD-127: Display uploaded attachments */}
      {(uploadedAttachments.length > 0 || uploadQueue.length > 0) && (
        <div className="mb-3 flex flex-wrap gap-2">
          {uploadedAttachments.map((att) => (
            <div
              key={att.attachment_id}
              className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs ${
                att.media_type === 'image'
                  ? 'border-info/20 bg-info/10 text-info/70'
                  : 'border-warning/20 bg-warning/10 text-amber-200'
              }`}
            >
              <span className="truncate max-w-[240px]">{att.filename}</span>
              <button
                type="button"
                className="ml-1 opacity-70 hover:opacity-100"
                onClick={() => setUploadedAttachments((prev) => prev.filter((a) => a.attachment_id !== att.attachment_id))}
                aria-label="Remove attachment"
              >
                ×
              </button>
            </div>
          ))}

          {uploadQueue.map((name) => (
            <div
              key={name}
              className="inline-flex items-center gap-2 rounded-full border border-primary/20 bg-primary/10 px-3 py-1 text-xs text-primary"
            >
              <span className="truncate max-w-[240px]">{name}</span>
              <span className="text-primary/70">(uploading…)</span>
            </div>
          ))}
        </div>
      )}

      {/* Large input box with everything inside - Incredible-style centered card */}
      <div
        className={[
          'relative w-full rounded-3xl border-2',
          'border-primary/20 focus-within:border-primary/40 focus-within:ring-2 focus-within:ring-primary/15',
          'transition-all',
        ].join(' ')}
      >
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
        <div className="flex items-center justify-between px-3 py-2 border-t border-transparent absolute bottom-0 left-0 right-0">
          {/* Left side: Attachment + Mic + Agent/Model Selector */}
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

            {/* PRD: Unified Agent-Chat System — the Agent Selector is the real
                routing control. PRD-180 S3 (F035): the placebo ModelSelector
                fallback was removed (the backend never read the chosen model —
                a control that does nothing corrodes trust in the real ones). */}
            {onAgentChange && (
              <AgentSelector
                selectedAgentId={selectedAgentId}
                onAgentChange={onAgentChange}
                onAgentData={setActiveAgent}
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
                    className="relative hover:scale-110 transition-all duration-220 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 rounded-lg"
                    title={`Click for ${tool.name} suggestions`}
                    aria-label={`Show suggestions for ${tool.name}`}
                  >
                    <ToolLogo
                      name={tool.name}
                      logo={tool.icon}
                      size={28}
                      showBackground={true}
                      className="bg-background/80 ring-1 ring-primary/30 border border-primary/20 rounded-lg"
                    />
                  </button>
                ))}
                {activeAgent.tools.length > 4 && (
                  <div className="w-8 h-8 rounded-lg bg-secondary/80 ring-1 ring-primary/30 flex items-center justify-center text-[10px] font-bold text-muted-foreground border border-primary/20 relative z-10">
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
                className="bg-destructive hover:bg-destructive/80 h-9 w-9 rounded-xl shadow-sm"
              >
                <StopCircle className="w-4 h-4" />
              </Button>
            ) : (
              <Button
                type="submit"
                disabled={!safeInput.trim()}
                size="icon"
                className="h-10 w-10 rounded-2xl disabled:opacity-50 bg-primary hover:bg-primary/90 transition-colors"
              >
                <Send className="w-4 h-4" />
              </Button>
            )}
          </div>
        </div>
      </div>

      {/* Usage Info */}
      {usage && usage.totalTokens && (
        <div className="flex items-center justify-between text-xs text-muted-foreground">
          <span>Tokens: {usage.totalTokens.toLocaleString()}</span>
          {usage.cost && <span>Cost: ${usage.cost.toFixed(4)}</span>}
        </div>
      )}

    </form>
  )
}
