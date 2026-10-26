'use client'

/**
 * The chat composer: the default chat's input, and Mission Mode's (components/chatbot/chat.tsx). PRD-264 EUAI-11a:
 * the box and every control but the voice button sit in the AI label's fence (AiLabelFence), so a theme that keeps
 * the page's label from being seen pauses typing and sending; the voice button (composer-voice.tsx) has its own
 * caption and spoken notice, and is never fenced.
 */
import { useState, useRef, useEffect, type RefObject } from 'react'
import { AiLabelFence } from '@/components/ai-label/ai-label-fence'
import { Textarea } from '@/components/ui/textarea'
import type { Agent } from './agent-selector'
import type { VisibilityType, AppUsage } from '@/types'

import { ComposerActions } from './composer-actions'
import { AttachmentPicker, AttachmentStrip, type ComposerAttachment, useComposerAttachments } from './composer-attachments'

/** The box grows with its text up to this height (its max-h-[200px] class), then scrolls. */
const MAX_TEXTAREA_HEIGHT_PX = 200

/** Large input box with everything inside - Incredible-style centered card. */
const BOX_CLASS = [
  'relative w-full rounded-3xl border-2',
  'border-primary/20 focus-within:border-primary/40 focus-within:ring-2 focus-within:ring-primary/15',
  'transition-all',
].join(' ')

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

/**
 * The message Send hands the chat. PRD-127: the attachments go as attachment_ids (not document:// URLs), and as file
 * parts for display (the filename chips in the message history).
 */
function composerMessage(text: string, attachments: ComposerAttachment[]) {
  return {
    role: 'user',
    content: text,
    attachment_ids: attachments.map((att) => att.attachment_id),
    parts: [
      ...attachments.map((att) => ({
        type: 'file',
        filename: att.filename,
        mediaType: att.media_type === 'image' ? 'image/*' : 'application/octet-stream',
        attachment_id: att.attachment_id,
      })),
      { type: 'text', text },
    ],
  }
}

/** The box grows with what is typed, and takes the focus back once a reply has finished streaming. */
function useTextareaFit(textareaRef: RefObject<HTMLTextAreaElement>, value: string, status: unknown): void {
  useEffect(() => {
    const textarea = textareaRef.current
    if (!textarea) return
    textarea.style.height = 'auto'
    textarea.style.height = `${Math.min(textarea.scrollHeight, MAX_TEXTAREA_HEIGHT_PX)}px`
  }, [textareaRef, value])

  // Refocus textarea when streaming finishes
  useEffect(() => {
    if (status !== 'streaming') {
      textareaRef.current?.focus()
    }
  }, [textareaRef, status])
}

export function MultimodalInput({
  status,
  stop,
  sendMessage,
  selectedAgentId,
  onAgentChange,
  usage,
  onToolIconClick,
}: MultimodalInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  // PRD-127: ephemeral attachments
  const attachments = useComposerAttachments()
  const [input, setInput] = useState('')
  const [activeAgent, setActiveAgent] = useState<Agent | null>(null)

  // Safe input with default
  const safeInput = input || ''
  useTextareaFit(textareaRef, safeInput, status)

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    const trimmedInput = safeInput.trim()
    if (!trimmedInput || status === 'streaming') return

    sendMessage(composerMessage(trimmedInput, attachments.uploaded))

    // Clear input and attachments
    setInput('')
    attachments.clear()

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
    <form onSubmit={handleSubmit} className="w-full">
      {/* PRD-127: Ephemeral attachment upload, and the uploaded attachments */}
      <AiLabelFence>
        <AttachmentPicker attachments={attachments} />
        <AttachmentStrip attachments={attachments} />
      </AiLabelFence>

      <div className={BOX_CLASS}>
        <AiLabelFence>
          <Textarea
            ref={textareaRef}
            value={safeInput}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Send a message..."
            className="min-h-[60px] max-h-[200px] w-full resize-none rounded-3xl bg-transparent border-0 px-4 pt-4 pb-14 text-base text-foreground placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-0 focus-visible:ring-offset-0"
            rows={1}
          />
        </AiLabelFence>

        {/* Bottom toolbar - inside the input */}
        <ComposerActions
          isStreaming={isStreaming}
          uploading={attachments.uploadQueue.length > 0}
          onAttach={() => attachments.fileInputRef.current?.click()}
          selectedAgentId={selectedAgentId}
          onAgentChange={onAgentChange}
          onAgentData={setActiveAgent}
          activeAgent={activeAgent}
          onToolIconClick={onToolIconClick}
          canSend={Boolean(safeInput.trim())}
          stop={stop}
        />
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
