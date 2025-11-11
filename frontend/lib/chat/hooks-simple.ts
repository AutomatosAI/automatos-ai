'use client'

import { useState, useCallback } from 'react'
import type { ChatMessage } from '@/types'

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
  const [isLoading, setIsLoading] = useState(false)
  const [status, setStatus] = useState<'idle' | 'streaming' | 'error'>('idle')
  
  const sendMessage = useCallback(async (message: any) => {
    const messageObj = typeof message === 'string' 
      ? { role: 'user', content: message }
      : message
    
    // Add user message to UI immediately
    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: messageObj.content || '',
      parts: [{ type: 'text', text: messageObj.content || '' }],
      createdAt: new Date(),
    }
    
    setMessages(prev => [...prev, userMessage])
    setIsLoading(true)
    setStatus('streaming')
    
    try {
      const token = typeof window !== 'undefined' ? localStorage.getItem('auth_token') : null
      const response = await fetch(`${API_BASE_URL}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { 'Authorization': `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({
          id,
          message: {
            role: 'user',
            parts: [{ type: 'text', text: messageObj.content }],
          },
          selectedChatModel: 'gpt-4',
          selectedVisibilityType: 'private',
        }),
      })
      
      if (!response.ok || !response.body) {
        throw new Error('Failed to fetch')
      }
      
      // Handle SSE stream
      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      
      let assistantMessage: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: '',
        parts: [],
        createdAt: new Date(),
      }
      
      setMessages(prev => [...prev, assistantMessage])
      
      let buffer = ''
      
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6)
            if (data === '[DONE]') continue
            
            try {
              const parsed = JSON.parse(data)
              
              if (parsed.type === 'text-delta' && parsed.delta) {
                assistantMessage.content += parsed.delta
                setMessages(prev => {
                  const updated = [...prev]
                  updated[updated.length - 1] = {
                    ...assistantMessage,
                    content: assistantMessage.content,
                  }
                  return updated
                })
              } else if (parsed.type === 'tool-data' && parsed.data) {
                // Attach structured tool results to the message
                console.log('[Chat Hook] Received tool-data:', parsed.data)
                console.log('[Chat Hook] Documents count:', parsed.data.documents?.length)
                console.log('[Chat Hook] Code snippets count:', parsed.data.code_snippets?.length)
                console.log('[Chat Hook] Database results count:', parsed.data.database_results?.length)
                
                assistantMessage = {
                  ...assistantMessage,
                  codeSnippets: parsed.data.code_snippets || assistantMessage.codeSnippets,
                  documents: parsed.data.documents || assistantMessage.documents,
                  database_results: parsed.data.database_results || assistantMessage.database_results,
                }
                console.log('[Chat Hook] Updated message:', assistantMessage)
                setMessages(prev => {
                  const updated = [...prev]
                  updated[updated.length - 1] = assistantMessage
                  return updated
                })
              } else if (parsed.type === 'data-usage') {
                if (onData) onData(parsed)
              } else if (parsed.type === 'done') {
                break
              }
            } catch (e) {
              console.warn('Failed to parse SSE:', data, e)
            }
          }
        }
      }
      
      setStatus('idle')
      setIsLoading(false)
      
    } catch (error) {
      console.error('Chat error:', error)
      setStatus('error')
      setIsLoading(false)
    }
  }, [id, onData])
  
  const reload = useCallback(() => {
    // Implement if needed
  }, [])
  
  const stop = useCallback(() => {
    setIsLoading(false)
    setStatus('idle')
  }, [])
  
  return {
    messages,
    setMessages,
    sendMessage,
    isLoading,
    status,
    stop,
    reload,
  }
}

