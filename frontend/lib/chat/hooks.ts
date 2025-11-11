'use client'

import { useState, useCallback } from 'react'
import type { ChatMessage, AppUsage } from '@/types'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || ''

export function useChat({
  id,
  initialMessages = [],
  onData,
}: {
  id: string
  initialMessages?: ChatMessage[]
  onData?: (data: any) => void
}) {
  const [messages, setMessages] = useState<ChatMessage[]>(initialMessages)
  const [usage, setUsage] = useState<AppUsage | undefined>()
  const [isLoading, setIsLoading] = useState(false)
  const [status, setStatus] = useState<'idle' | 'streaming' | 'error'>('idle')
  let abortController: AbortController | null = null

  const stop = useCallback(() => {
    if (abortController) {
      abortController.abort()
      setIsLoading(false)
      setStatus('idle')
    }
  }, [])

  const reload = useCallback(() => {
    // Remove last assistant message and resend last user message
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

      // Add user message immediately
      setMessages(prev => [...prev, userMessage])
      setIsLoading(true)
      setStatus('streaming')

      // Create empty assistant message that we'll update
      const assistantMessageId = crypto.randomUUID()
      const assistantMessage: ChatMessage = {
        id: assistantMessageId,
        role: 'assistant',
        content: '',
        parts: [],
      }

      setMessages(prev => [...prev, assistantMessage])

      try {
        abortController = new AbortController()
        const token = typeof window !== 'undefined' ? localStorage.getItem('token') : null

        const response = await fetch(`${API_BASE_URL}/api/chat`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            ...(token ? { 'Authorization': `Bearer ${token}` } : {}),
          },
          body: JSON.stringify({
            id: id || '',
            message: userMessage,
            selectedChatModel: 'gpt-4',
            selectedVisibilityType: 'private',
          }),
          signal: abortController.signal,
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
            if (!line.trim() || !line.startsWith('data: ')) continue

            const data = line.slice(6).trim()
            if (data === '[DONE]') continue

            try {
              const parsed = JSON.parse(data)
              console.log('[Chat] SSE chunk:', parsed)

              if (parsed.type === 'text-delta' && parsed.delta) {
                accumulatedContent += parsed.delta
                
                // Update the assistant message with accumulated content
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
              } else if (parsed.type === 'data-usage') {
                setUsage(parsed.data)
                if (onData) onData(parsed)
              }
            } catch (e) {
              console.warn('[Chat] Failed to parse SSE line:', line, e)
            }
          }
        }

        setIsLoading(false)
        setStatus('idle')
      } catch (error: any) {
        if (error.name !== 'AbortError') {
          console.error('[Chat] Stream error:', error)
          setStatus('error')
        }
        setIsLoading(false)
      }
    },
    [id, isLoading, onData]
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
