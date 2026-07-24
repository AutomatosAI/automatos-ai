'use client'

import { useState, useCallback, useEffect, useRef } from 'react'
import { useAuth } from '@clerk/nextjs'
import type { ChatMessage, AppUsage, ToolCall, RoutingInfo } from '@/types'
import type { PageContext } from '@/lib/page-context'
import { toast } from 'sonner'

export function useChat({
  id,
  initialMessages = [],
  selectedAgentId,
  missionMode = false,
  planMode = false,
  pageContext,
  onData,
  onChatIdUpdate,
  onRoutingDecision,
}: {
  id: string
  initialMessages?: ChatMessage[]
  selectedAgentId?: number | null
  missionMode?: boolean
  planMode?: boolean
  // PRD-221 S5 (extends PRD-220): structured page context — where the user is
  // plus what they're looking at (references only). Sent as request context,
  // injected prompt-side by the backend — never stored in the message or title.
  pageContext?: PageContext
  onData?: (data: any) => void
  onChatIdUpdate?: (chatId: string) => void
  onRoutingDecision?: (info: RoutingInfo) => void
}) {
  const { getToken, isLoaded } = useAuth()
  const [messages, setMessages] = useState<ChatMessage[]>(initialMessages)
  const [usage, setUsage] = useState<AppUsage | undefined>()
  const [isLoading, setIsLoading] = useState(false)
  const [status, setStatus] = useState<'idle' | 'streaming' | 'error'>('idle')
  const [chatId, setChatId] = useState(id)
  const abortControllerRef = useRef<AbortController | null>(null)

  // PRD-207: the id prop can move mid-mount (a live call binds the screen to
  // its thread) — follow it so the chat_changed merge listener and sends
  // target the conversation actually on screen.
  useEffect(() => {
    setChatId(id)
  }, [id])

  const stop = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      setIsLoading(false)
      setStatus('idle')
    }
  }, [])

  // PRD-205 S7: a background producer posted into a chat (watcher verdict,
  // scheduled-task output). The SSE lane fans it out as a window event; when
  // it targets THIS conversation, refetch and merge missing messages by id.
  // Append-only: an in-flight streaming placeholder has an id the server
  // doesn't know, so it is never clobbered.
  useEffect(() => {
    if (typeof window === 'undefined') return
    const onChatChanged = (event: Event) => {
      const detail = (event as CustomEvent).detail as { chat_id?: string } | undefined
      if (!detail?.chat_id || !chatId || detail.chat_id !== chatId) return
      void (async () => {
        try {
          const { getChatMessages } = await import('@/lib/chat/api')
          const serverMessages = await getChatMessages(chatId)
          setMessages((prev) => {
            const known = new Set(prev.map((m) => m.id))
            const missing = serverMessages.filter((m) => !known.has(m.id))
            return missing.length > 0 ? [...prev, ...missing] : prev
          })
        } catch {
          // Best-effort: the message still appears on next open/reload.
        }
      })()
    }
    window.addEventListener('automatos:chat-changed', onChatChanged)
    return () => window.removeEventListener('automatos:chat-changed', onChatChanged)
  }, [chatId])

  const reload = useCallback(() => {
    const lastUserMessageIndex = messages.findLastIndex(m => m.role === 'user')
    if (lastUserMessageIndex >= 0) {
      const lastUserMessage = messages[lastUserMessageIndex]
      setMessages(messages.slice(0, lastUserMessageIndex))
      sendMessage(lastUserMessage.content || '')
    }
  }, [messages])

  const sendMessage = useCallback(
    async (message: any) => {
      if (isLoading) return

      const messageObj = typeof message === 'string'
        ? { role: 'user', content: message }
        : message

      const userMessage: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'user',
        content: messageObj.content || '',
        parts: messageObj.parts || [{ type: 'text', text: messageObj.content || '' }],
      }

      setMessages(prev => [...prev, userMessage])
      setIsLoading(true)
      setStatus('streaming')

      const assistantMessageId = crypto.randomUUID()
      const assistantMessage: ChatMessage = {
        id: assistantMessageId,
        role: 'assistant',
        content: '',
        parts: [],
      }

      setMessages(prev => [...prev, assistantMessage])

      try {
        abortControllerRef.current = new AbortController()
        const token = isLoaded ? await getToken() : null
        
        // API key is handled server-side in /api/chat route to avoid exposing secrets in client bundle
        // Only use localStorage API key if explicitly set by user (non-sensitive identifier)
        const apiKey = typeof window !== 'undefined' 
          ? localStorage.getItem('api_key')
          : null
        
        const outgoingParts =
          Array.isArray(messageObj.parts) && messageObj.parts.length > 0
            ? messageObj.parts
            : [{ type: 'text', text: messageObj.content || '' }]

        const response = await fetch('/api/chat', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            ...(token ? { 'Authorization': `Bearer ${token}` } : {}),
            ...(apiKey ? { 'x-api-key': apiKey } : {}),
            // Ensure backend gets the correct workspace context (prevents dev fallback UUID)
            ...(typeof window !== 'undefined' && localStorage.getItem('last_active_workspace')
              ? { 'X-Workspace-ID': localStorage.getItem('last_active_workspace') as string }
              : {}),
          },
          body: JSON.stringify({
            id: chatId || '',
            message: {
              role: 'user',
              parts: outgoingParts,
            },
            // PRD: Unified Agent-Chat System — send agentId when one is selected.
            // PRD-180 S3 (F035): the no-agent branch no longer sends a client
            // model override (the backend never read it). With no agent, the
            // model resolves via the Auto tier server-side.
            ...(selectedAgentId ? { agentId: selectedAgentId } : {}),
            selectedVisibilityType: 'private',
            // PRD-82A: Mission mode — conversational mission planning
            ...(missionMode ? { missionMode: true } : {}),
            // Plan mode — research and strategy, no execution
            ...(planMode ? { planMode: true } : {}),
            // PRD-220: page context for the widget (prompt-only server-side)
            ...(pageContext ? { context: pageContext } : {}),
          }),
          signal: abortControllerRef.current.signal,
        })

        if (!response.ok) {
          let errorText = ''
          try {
            errorText = await response.text()
          } catch (e) {
            // ignore
          }
          // Remove the empty assistant placeholder to avoid "blank bot bubbles"
          setMessages((prev) => prev.filter((m) => m.id !== assistantMessageId))
          setIsLoading(false)
          setStatus('error')
          toast.error(`Chat request failed (${response.status})${errorText ? `: ${errorText}` : ''}`)
          return
        }

        // Extract routing decision from response headers (set by universal router)
        const routingAgentId = response.headers.get('x-routing-agent-id')
        const routingConfidence = response.headers.get('x-routing-confidence')
        const routingType = response.headers.get('x-routing-type')
        const routingReasoning = response.headers.get('x-routing-reasoning')
        const routingRequestId = response.headers.get('x-routing-request-id')

        let routingInfo: RoutingInfo | undefined
        if (routingAgentId && routingType) {
          routingInfo = {
            requestId: routingRequestId || undefined,
            agentId: parseInt(routingAgentId, 10),
            confidence: routingConfidence ? parseFloat(routingConfidence) : 0,
            routeType: routingType,
            reasoning: routingReasoning || '',
          }
          // Attach routing info to the assistant message immediately
          setMessages(prev =>
            prev.map(m =>
              m.id === assistantMessageId ? { ...m, routingInfo } : m
            )
          )
          if (onRoutingDecision) onRoutingDecision(routingInfo)
        }

        const reader = response.body?.getReader()
        const decoder = new TextDecoder()
        let buffer = ''
        let accumulatedContent = ''

        const upsertToolCall = (current: ToolCall[] | undefined, next: ToolCall): ToolCall[] => {
          const list = current ? [...current] : []
          const idx = list.findIndex((t) => t.toolCallId === next.toolCallId)
          if (idx >= 0) {
            list[idx] = { ...list[idx], ...next }
            return list
          }
          return [...list, next]
        }

        while (reader) {
          const { done, value } = await reader.read()
          if (done) break

          buffer += decoder.decode(value, { stream: true })
          const lines = buffer.split('\n')
          buffer = lines.pop() || ''

          for (const line of lines) {
            if (!line.trim()) continue

            // AI SDK Data Stream format
            if (line.startsWith('0:')) {
              // Text chunk
              try {
                const text = JSON.parse(line.slice(2))
                accumulatedContent += text

                setMessages(prev =>
                  prev.map(m =>
                    m.id === assistantMessageId
                      ? {
                        ...m,
                        content: accumulatedContent,
                        parts: [{ type: 'text', text: accumulatedContent }],
                      }
                      : m
                  )
                )
              } catch (e) {
                // Skip parse errors
              }
            } else if (line.startsWith('d:')) {
              // Data event
              try {
                const data = JSON.parse(line.slice(2))

                // Handle chat-id event - critical for conversation continuity
                if (data.type === 'chat-id' && data.chatId) {
                  setChatId(data.chatId)
                  if (onChatIdUpdate) onChatIdUpdate(data.chatId)
                } else if (data.type === 'tool-start' && data.data?.toolCallId) {
                  const now = new Date().toISOString()
                  const toolCall: ToolCall = {
                    toolCallId: data.data.toolCallId,
                    toolName: data.data.toolName || 'tool',
                    state: 'running',
                    input: data.data.input,
                    startedAt: now,
                  }

                  setMessages((prev) =>
                    prev.map((m) =>
                      m.id === assistantMessageId
                        ? {
                          ...m,
                          toolCalls: upsertToolCall(m.toolCalls, toolCall),
                        }
                        : m
                    )
                  )

                  if (onData) onData({ type: 'tool-start', data: data.data })
                } else if (data.type === 'tool-end' && data.data?.toolCallId) {
                  const now = new Date().toISOString()
                  const toolCall: ToolCall = {
                    toolCallId: data.data.toolCallId,
                    toolName: data.data.toolName || 'tool',
                    state: data.data.success ? 'completed' : 'error',
                    error: data.data.error,
                    durationMs: data.data.durationMs,
                    endedAt: now,
                  }

                  setMessages((prev) =>
                    prev.map((m) =>
                      m.id === assistantMessageId
                        ? {
                          ...m,
                          toolCalls: upsertToolCall(m.toolCalls, toolCall),
                        }
                        : m
                    )
                  )

                  if (onData) onData({ type: 'tool-end', data: data.data })
                } else if (data.type === 'tool-data' && data.data) {
                  setMessages(prev =>
                    prev.map(m =>
                      m.id === assistantMessageId
                        ? {
                          ...m,
                          database_results: data.data.database_results || m.database_results,
                          documents: data.data.documents || m.documents,
                          // Convert snake_case from backend to camelCase for frontend
                          codeSnippets: data.data.code_snippets || m.codeSnippets,
                        }
                        : m
                    )
                  )
                  if (onData) onData({ type: 'tool-data', data: data.data })
                } else if (data.type === 'usage' && data.data) {
                  setUsage({
                    promptTokens: data.data.promptTokens || 0,
                    completionTokens: data.data.completionTokens || 0,
                    totalTokens: data.data.totalTokens || 0,
                  })
                  if (onData) onData({ type: 'data-usage', data: data.data })
                }
                // PRD-67: Forward agent-info (including CTO mode) to onData
                else if (data.type === 'agent-info' && data.agent) {
                  if (onData) onData({ type: 'agent-info', data: data.agent })
                }
                // US-015: Widget SSE events — forward to onData for workspace store
                else if (data.type === 'memory-injected' && data.data) {
                  if (onData) onData({ type: 'memory-injected', data: data.data })
                } else if (data.type === 'memory-stored' && data.data) {
                  if (onData) onData({ type: 'memory-stored', data: data.data })
                } else if (data.type === 'workflow-update' && data.data) {
                  if (onData) onData({ type: 'workflow-update', data: data.data })
                }
                // PRD-125 Phase 1: Forward mission-suggestion to onData for chat card
                else if (data.type === 'mission-suggestion' && data.data) {
                  if (onData) onData({ type: 'mission-suggestion', data: data.data })
                }
              } catch (e) {
                // Skip parse errors
              }
            } else if (line.startsWith('data:')) {
              // Legacy SSE fallback (some backends send `data: {json}\n\n`)
              try {
                const payload = JSON.parse(line.replace(/^data:\s*/, ''))

                if (payload.type === 'text-delta' && payload.delta) {
                  accumulatedContent += payload.delta
                  setMessages((prev) =>
                    prev.map((m) =>
                      m.id === assistantMessageId
                        ? {
                          ...m,
                          content: accumulatedContent,
                          parts: [{ type: 'text', text: accumulatedContent }],
                        }
                        : m
                    )
                  )
                } else if (payload.type === 'tool-data' && payload.data) {
                  setMessages((prev) =>
                    prev.map((m) =>
                      m.id === assistantMessageId
                        ? {
                          ...m,
                          database_results: payload.data.database_results || m.database_results,
                          documents: payload.data.documents || m.documents,
                          codeSnippets: payload.data.code_snippets || m.codeSnippets,
                        }
                        : m
                    )
                  )
                  if (onData) onData({ type: 'tool-data', data: payload.data })
                } else if (payload.type === 'data-usage' && payload.data) {
                  setUsage({
                    promptTokens: payload.data.promptTokens || 0,
                    completionTokens: payload.data.completionTokens || 0,
                    totalTokens: payload.data.totalTokens || 0,
                  })
                  if (onData) onData({ type: 'data-usage', data: payload.data })
                } else if (payload.type === 'mission-suggestion' && payload.data) {
                  if (onData) onData({ type: 'mission-suggestion', data: payload.data })
                } else if (payload.type === 'error') {
                  setStatus('error')
                } else if (payload.type === 'done') {
                  setStatus('idle')
                }
              } catch (e) {
                // Skip parse errors
              }
            } else if (line.startsWith('e:')) {
              // Error
              console.error('[Chat] Error:', line.slice(2))
              setStatus('error')
            }
          }
        }

        setIsLoading(false)
        setStatus('idle')
      } catch (error: any) {
        if (error.name !== 'AbortError') {
          console.error('[Chat] Error:', error)
          setStatus('error')
          toast.error(error?.message || 'Chat failed')
        }
        // Remove the empty assistant placeholder to avoid "blank bot bubbles"
        setMessages((prev) => prev.filter((m) => m.id !== assistantMessageId))
        setIsLoading(false)
      }
    },
    [chatId, isLoading, selectedAgentId, missionMode, planMode, pageContext, onData, onChatIdUpdate, onRoutingDecision]
  )

  return {
    messages,
    setMessages,
    sendMessage,
    reload,
    status,
    stop,
    isLoading,
    usage,
  }
}
