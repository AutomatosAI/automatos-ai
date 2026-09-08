'use client'

import { useState, useCallback, useEffect, useRef } from 'react'
import { useAuth } from '@/lib/auth-hooks'
import type { ChatMessage, AppUsage, ToolCall, RoutingInfo } from '@/types'
import type { PageContext } from '@/lib/page-context'
import { TRIAL_EXHAUSTED_CODE } from '@/lib/trial'
import { completeRunningToolCalls, upsertTaskCard, upsertToolCall } from '@/lib/chat/tool-calls'
import { toast } from 'sonner'

/** PRD-237 S7: the client-side placeholder shown while the server finishes a turn. */
export const AWAITING_REPLY_ID = 'awaiting-reply'
/** Give up waiting for a detached reply after this long (the turn itself is capped server-side). */
const AWAITING_REPLY_GUARD_MS = 5 * 60_000
/** PRD-238 S4: progress lines kept per message (the newest win). */
const PROGRESS_LINES_KEPT = 6

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
  initialAwaitingReply = false,
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
  // PRD-237 S7: the server is still producing a reply for this chat (the page
  // reloaded mid-turn) — show the typing state until it merges in.
  initialAwaitingReply?: boolean
}) {
  const { getToken, isLoaded } = useAuth()
  const [messages, setMessages] = useState<ChatMessage[]>(initialMessages)
  const [usage, setUsage] = useState<AppUsage | undefined>()
  const [isLoading, setIsLoading] = useState(false)
  const [status, setStatus] = useState<'idle' | 'streaming' | 'error'>('idle')
  // PRD-222 US-014: a stable error code the surfaces can render deterministically.
  // Set to 'trial_exhausted' when a send is blocked by a spent trial (pre-stream
  // non-ok body or a mid-stream error line); reset at the start of every send.
  const [errorCode, setErrorCode] = useState<string | null>(null)
  const [chatId, setChatId] = useState(id)
  const abortControllerRef = useRef<AbortController | null>(null)

  // PRD-237 S7: a reload mid-reply. The turn finishes detached on the server;
  // hold a typing placeholder until chat_changed brings the reply (or a guard
  // gives up). Mirrored in a ref so the merge listener needn't resubscribe.
  const [awaitingReply, setAwaitingReply] = useState(initialAwaitingReply)
  const awaitingRef = useRef(initialAwaitingReply)
  useEffect(() => {
    awaitingRef.current = awaitingReply
  }, [awaitingReply])
  useEffect(() => {
    if (!initialAwaitingReply) return
    setMessages((prev) =>
      prev.some((m) => m.id === AWAITING_REPLY_ID)
        ? prev
        : [...prev, { id: AWAITING_REPLY_ID, role: 'assistant', content: '', parts: [] } as ChatMessage],
    )
    const guard = setTimeout(() => {
      awaitingRef.current = false
      setAwaitingReply(false)
      setMessages((prev) => prev.filter((m) => m.id !== AWAITING_REPLY_ID))
    }, AWAITING_REPLY_GUARD_MS)
    return () => clearTimeout(guard)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // PRD-237 S7: leaving the surface (route change, tab switch) drops the
  // connection on purpose — the server finishes the turn detached and the
  // reply arrives via chat_changed. Stop is the only intentional cancel.
  useEffect(() => () => abortControllerRef.current?.abort(), [])

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
    // PRD-237 S7: the turn no longer dies with the connection, so Stop has to
    // say so — the cancel reaches whichever worker holds the turn.
    if (chatId) {
      void import('@/lib/chat/api')
        .then(({ cancelChatTurn }) => cancelChatTurn(chatId))
        .catch(() => {})
    }
  }, [chatId])

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
          const last = serverMessages[serverMessages.length - 1]
          // PRD-237 S7: the awaited (detached) reply landed — the server list
          // is the truth now; the placeholder goes with it.
          if (awaitingRef.current && last?.role === 'assistant') {
            awaitingRef.current = false
            setAwaitingReply(false)
            setMessages(serverMessages)
            return
          }
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
      if (isLoading || awaitingRef.current) return

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
      setErrorCode(null) // clear any prior block before this attempt

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
            ...(token ? { ...(token ? { Authorization: `Bearer ${token}` } : {}),} : {}),
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
          // PRD-222 US-014: a spent trial is blocked with a typed code — surface
          // it so the exhausted banner appears immediately (before the snapshot
          // refreshes). The gate can trip pre-stream (this non-ok body).
          if (errorText.includes(TRIAL_EXHAUSTED_CODE)) setErrorCode(TRIAL_EXHAUSTED_CODE)
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
                    // PRD-238 S3: a skipped (de-duplicated) call closes its line
                    // as done-without-running, never as an error.
                    state: data.data.success || data.data.skipped ? 'completed' : 'error',
                    error: data.data.error,
                    durationMs: data.data.durationMs,
                    summary: data.data.summary,
                    skipped: Boolean(data.data.skipped),
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
                          // PRD-238 S6: a ticket card (filed / checked / awaited), one per ticket id
                          taskCards: data.data.task_card
                            ? upsertTaskCard(m.taskCards, data.data.task_card)
                            : m.taskCards,
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
                // PRD-238 S1: the thinking channel — shown live, never as the answer.
                else if (data.type === 'reasoning' && typeof data.data?.delta === 'string') {
                  const delta = data.data.delta as string
                  setMessages((prev) =>
                    prev.map((m) =>
                      m.id === assistantMessageId ? { ...m, reasoning: (m.reasoning ?? '') + delta } : m
                    )
                  )
                }
                // PRD-238 S4: a progress line from inside a long-running tool call.
                else if (data.type === 'progress' && typeof data.data?.text === 'string') {
                  const line = data.data.text as string
                  setMessages((prev) =>
                    prev.map((m) =>
                      m.id === assistantMessageId
                        ? { ...m, progress: [...(m.progress ?? []).slice(-(PROGRESS_LINES_KEPT - 1)), line] }
                        : m
                    )
                  )
                }
                // PRD-238 S3: the turn is over — nothing may keep spinning.
                else if (data.type === 'finish') {
                  const endedAt = new Date().toISOString()
                  setMessages((prev) =>
                    prev.map((m) =>
                      m.id === assistantMessageId
                        ? { ...m, toolCalls: completeRunningToolCalls(m.toolCalls, endedAt) }
                        : m
                    )
                  )
                }
                // PRD-238 S3: a cap ended the turn — say so instead of going quiet.
                else if (data.type === 'limit_reached' && data.data?.message) {
                  const limit = {
                    limit: String(data.data.limit ?? ''),
                    value: Number(data.data.value ?? 0),
                    message: String(data.data.message),
                  }
                  setMessages((prev) =>
                    prev.map((m) => (m.id === assistantMessageId ? { ...m, limitReached: limit } : m))
                  )
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
                  // PRD-222 US-014: mid-stream trial block carries the typed code.
                  if (
                    payload.error_code === TRIAL_EXHAUSTED_CODE ||
                    payload.code === TRIAL_EXHAUSTED_CODE ||
                    (typeof payload.error === 'string' &&
                      payload.error.includes(TRIAL_EXHAUSTED_CODE))
                  ) {
                    setErrorCode(TRIAL_EXHAUSTED_CODE)
                  }
                } else if (payload.type === 'done') {
                  setStatus('idle')
                }
              } catch (e) {
                // Skip parse errors
              }
            } else if (line.startsWith('e:')) {
              // Error
              const errLine = line.slice(2)
              console.error('[Chat] Error:', errLine)
              setStatus('error')
              // PRD-222 US-014: mid-stream trial block → typed code for the banner.
              if (errLine.includes(TRIAL_EXHAUSTED_CODE)) setErrorCode(TRIAL_EXHAUSTED_CODE)
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
    // PRD-237 S7: an awaited detached reply reads as streaming to the surfaces.
    status: awaitingReply ? 'streaming' : status,
    stop,
    isLoading: isLoading || awaitingReply,
    awaitingReply,
    usage,
    // PRD-222 US-014: 'trial_exhausted' when the last send was blocked by a
    // spent trial; null otherwise. Drives the deterministic exhausted banner.
    errorCode,
  }
}
