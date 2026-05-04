'use client'

import { useState, useRef, useCallback, useEffect } from 'react'
import { Send, StopCircle, Paperclip, Phone } from 'lucide-react'
import { useAuth } from '@clerk/nextjs'
import { AnimatePresence } from 'framer-motion'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { ModelSelector } from './model-selector'
import { AgentSelector, type Agent } from './agent-selector'
import { ToolLogo } from '@/components/ui/tool-logo'
import { VoiceMicButton } from '@/components/voice/VoiceMicButton'
import { VoiceRecordingIndicator } from '@/components/voice/VoiceRecordingIndicator'
import { useVoiceRecorder } from '@/hooks/use-voice-recorder'
import { sendVoiceMessage, checkVoiceHealth, getVoiceAudioUrl } from '@/lib/voice-client'
import { VoiceCallPanel } from '@/components/voice/VoiceCallPanel'
import type { VisibilityType, AppUsage } from '@/types'
import { apiClient } from '@/lib/api-client'
import { toast } from 'sonner'
import { useWorkspace } from '@/components/workspace-provider'

export interface MultimodalInputProps {
  chatId: string
  status: any
  stop: () => void
  sendMessage: (message: any) => void
  setMessages?: React.Dispatch<React.SetStateAction<any[]>>
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
  setMessages,
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
  // PRD-127: Switch to ephemeral attachments
  const [uploadedAttachments, setUploadedAttachments] = useState<Array<{
    attachment_id: string
    filename: string
    media_type: 'image' | 'document'
  }>>([])
  const [input, setInput] = useState('')
  const [activeAgent, setActiveAgent] = useState<Agent | null>(null)
  const [voiceEnabled, setVoiceEnabled] = useState(false)
  const [showCallPanel, setShowCallPanel] = useState(false)
  const { getToken } = useAuth()
  const { workspace } = useWorkspace()

  // Voice recording — hook lifted here so both mic button and indicator can control it
  const handleVoiceComplete = useCallback(
    async (blob: Blob, durationMs: number) => {
      try {
        const token = await getToken()
        const response = await sendVoiceMessage(blob, chatId, {
          agentId: selectedAgentId ?? undefined,
          responseFormat: 'both',
          authToken: token,
        })

        // The voice endpoint already ran the full pipeline (STT → agent → TTS)
        // and saved messages to the DB. Inject both messages directly into the
        // chat UI — do NOT call sendMessage() which would trigger a second agent call.
        if (setMessages) {
          const userMsg = {
            id: crypto.randomUUID(),
            role: 'user' as const,
            content: response.transcript,
            parts: [{
              type: 'voice' as const,
              transcript: response.transcript,
              audioUrl: undefined,
              durationMs,
            }],
          }
          const assistantMsg = {
            id: response.message_id || crypto.randomUUID(),
            role: 'assistant' as const,
            content: response.response_text,
            parts: [
              { type: 'text' as const, text: response.response_text },
              ...((response.audio_base64 || response.audio_url) ? [{
                type: 'voice' as const,
                transcript: response.response_text,
                audioUrl: getVoiceAudioUrl(response.message_id),
                audioBase64: response.audio_base64 || undefined,
              }] : []),
            ],
          }
          setMessages(prev => [...prev, userMsg, assistantMsg])
        } else {
          // Fallback if setMessages not available — send transcript through chat
          sendMessage({
            role: 'user',
            content: response.transcript,
            parts: [{ type: 'voice', transcript: response.transcript, durationMs }],
          })
        }
      } catch (err: any) {
        toast.error(err?.message || 'Voice message failed')
      }
    },
    [chatId, selectedAgentId, sendMessage, setMessages, getToken]
  )

  const voiceRecorder = useVoiceRecorder({
    maxDurationMs: 120_000,
    onRecordingComplete: handleVoiceComplete,
  })

  // Check voice service availability on mount
  useEffect(() => {
    checkVoiceHealth()
      .then((health) => {
        setVoiceEnabled(health.voice_enabled && health.voice_service_healthy)
      })
      .catch(() => {
        setVoiceEnabled(false)
      })
  }, [])

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
          'relative w-full rounded-3xl border-2',
          voiceRecorder.state === 'recording'
            ? 'border-destructive/30 ring-2 ring-destructive/15'
            : 'border-orange-500/20 focus-within:border-orange-500/40 focus-within:ring-2 focus-within:ring-orange-500/15',
          'transition-all shadow-[0_0_60px_rgba(249,115,22,0.08)]',
        ].join(' ')}
      >
        {/* Textarea or Recording Indicator */}
        <AnimatePresence mode="wait">
          {voiceRecorder.state === 'recording' ? (
            <VoiceRecordingIndicator
              key="recording"
              durationMs={voiceRecorder.durationMs}
              onStop={voiceRecorder.stopRecording}
              onCancel={voiceRecorder.cancelRecording}
            />
          ) : (
            <Textarea
              ref={textareaRef}
              value={safeInput}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={voiceRecorder.state === 'processing' ? 'Processing voice...' : 'Send a message...'}
              disabled={voiceRecorder.state === 'processing'}
              className="min-h-[60px] max-h-[200px] w-full resize-none rounded-3xl bg-transparent border-0 px-4 pt-4 pb-14 text-base text-foreground placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-0 focus-visible:ring-offset-0"
              rows={1}
            />
          )}
        </AnimatePresence>

        {/* Bottom toolbar - inside the input */}
        <div className={[
          'flex items-center justify-between px-3 py-2 border-t border-transparent',
          voiceRecorder.state === 'recording' ? 'hidden' : 'absolute bottom-0 left-0 right-0',
        ].join(' ')}>
          {/* Left side: Attachment + Mic + Agent/Model Selector */}
          <div className="flex items-center gap-2">
            <Button
              type="button"
              variant="ghost"
              size="sm"
              className="h-8 w-8 p-0 text-muted-foreground hover:text-foreground"
              disabled={isStreaming || uploadQueue.length > 0 || voiceRecorder.state !== 'idle'}
              onClick={() => fileInputRef.current?.click()}
            >
              <Paperclip className="w-4 h-4" />
            </Button>

            {/* Voice Mic Button */}
            {voiceEnabled && (
              <VoiceMicButton
                state={voiceRecorder.state}
                durationMs={voiceRecorder.durationMs}
                onStartRecording={voiceRecorder.startRecording}
                onStopRecording={voiceRecorder.stopRecording}
                error={voiceRecorder.error}
                disabled={isStreaming || voiceRecorder.state === 'processing'}
              />
            )}

            {/* Live Voice Call Button (Phase 3) — disabled for pilot, WebSocket needs debugging */}
            {/* {voiceEnabled && (
              <Button
                type="button"
                variant="ghost"
                size="sm"
                className={[
                  'h-8 w-8 p-0',
                  showCallPanel
                    ? 'text-success hover:text-success'
                    : 'text-muted-foreground hover:text-foreground',
                ].join(' ')}
                disabled={isStreaming || voiceRecorder.state !== 'idle'}
                onClick={() => setShowCallPanel((prev) => !prev)}
                title="Live voice call"
              >
                <Phone className="w-4 h-4" />
              </Button>
            )} */}

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
                    className="relative hover:scale-110 transition-all duration-220 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 rounded-lg"
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
                  <div className="w-8 h-8 rounded-lg bg-secondary/80 ring-1 ring-orange-500/30 flex items-center justify-center text-[10px] font-bold text-muted-foreground border border-orange-500/20 relative z-10">
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
        <div className="flex items-center justify-between text-xs text-muted-foreground">
          <span>Tokens: {usage.totalTokens.toLocaleString()}</span>
          {usage.cost && <span>Cost: ${usage.cost.toFixed(4)}</span>}
        </div>
      )}

      {/* Live Voice Call Panel (Phase 3) — disabled for pilot */}
      {/* <AnimatePresence>
        {showCallPanel && workspace?.id && (
          <VoiceCallPanel
            workspaceId={workspace.id}
            agentId={selectedAgentId}
            conversationId={chatId}
            agentName={activeAgent?.name ?? 'Auto'}
            onClose={() => setShowCallPanel(false)}
          />
        )}
      </AnimatePresence> */}
    </form>
  )
}
