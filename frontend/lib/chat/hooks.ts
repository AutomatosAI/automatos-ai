'use client'

import { useState, useCallback, useRef } from 'react'
import type { ChatMessage, AppUsage } from '@/types'

export function useChat({
  id,
  initialMessages = [],
  selectedModelId = 'gpt-4',
  onData,
  onChatIdUpdate,
}: {
  id: string
  initialMessages?: ChatMessage[]
  selectedModelId?: string
  onData?: (data: any) => void
  onChatIdUpdate?: (chatId: string) => void
}) {
  const [messages, setMessages] = useState<ChatMessage[]>(initialMessages)
  const [usage, setUsage] = useState<AppUsage | undefined>()
  const [isLoading, setIsLoading] = useState(false)
  const [status, setStatus] = useState<'idle' | 'streaming' | 'error'>('idle')
  const [chatId, setChatId] = useState(id)
  const abortControllerRef = useRef<AbortController | null>(null)

  const stop = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      setIsLoading(false)
      setStatus('idle')
    }
  }, [])

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
        const token = typeof window !== 'undefined' ? localStorage.getItem('token') : null

        const response = await fetch('/api/chat', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            ...(token ? { 'Authorization': `Bearer ${token}` } : {}),
          },
          body: JSON.stringify({
            id: chatId || '',
            message: {
              role: 'user',
              parts: [{ type: 'text', text: messageObj.content || '' }],
            },
            selectedChatModel: selectedModelId,
            selectedVisibilityType: 'private',
          }),
          signal: abortControllerRef.current.signal,
        })

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`)
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
        }
        setIsLoading(false)
      }
    },
    [chatId, isLoading, selectedModelId, onData, onChatIdUpdate]
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
