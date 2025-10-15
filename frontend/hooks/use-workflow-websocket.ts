'use client'

import { useEffect, useRef, useState, useCallback } from 'react'

interface WebSocketMessage {
  type: string
  data?: any
  execution_id?: number
  workflow_id?: number
  message?: string
  timestamp?: string
}

interface UseWorkflowWebSocketProps {
  workflowId?: number
  executionId?: number
  onMessage?: (message: WebSocketMessage) => void
  autoConnect?: boolean
}

export function useWorkflowWebSocket({
  workflowId,
  executionId,
  onMessage,
  autoConnect = true
}: UseWorkflowWebSocketProps) {
  const [isConnected, setIsConnected] = useState(false)
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null)
  const [error, setError] = useState<string | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>()
  const reconnectAttemptsRef = useRef(0)
  const maxReconnectAttempts = 10

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      console.log('WebSocket already connected')
      return
    }

    // Redis WebSocket requires both workflowId and executionId
    if (!workflowId || !executionId) {
      console.warn('⚠️ Cannot connect WebSocket: workflowId and executionId required', {
        workflowId,
        executionId
      })
      setError('Missing workflowId or executionId')
      return
    }

    try {
      // Determine WebSocket URL based on environment
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://206.81.0.227:8000'
      const url = new URL(apiUrl)
      
      // Use ws:// for http and wss:// for https
      const protocol = url.protocol === 'https:' ? 'wss:' : 'ws:'
      const host = url.host
      
      // Redis WebSocket endpoint: /ws/{workflow_id}/{execution_id}
      const wsUrl = `${protocol}//${host}/ws/${workflowId}/${executionId}`
      
      console.log('🔌 Connecting to Redis WebSocket:', wsUrl)
      console.log('📍 Workflow ID:', workflowId, 'Execution ID:', executionId)
      console.log('🌐 Protocol:', protocol, 'Host:', host)
      
      const ws = new WebSocket(wsUrl)

      ws.onopen = () => {
        console.log('✅ Redis WebSocket connected - subscribed to workflow:', workflowId, 'execution:', executionId)
        setIsConnected(true)
        setError(null)
        reconnectAttemptsRef.current = 0
      }

      ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data)
          console.log('📨 WebSocket message:', message)
          
          setLastMessage(message)
          
          // Call custom handler
          if (onMessage) {
            onMessage(message)
          }
        } catch (err) {
          console.error('Failed to parse WebSocket message:', err)
        }
      }

      ws.onerror = (event) => {
        console.error('❌ WebSocket error:', event)
        setError('WebSocket connection error')
      }

      ws.onclose = (event) => {
        console.log('🔌 WebSocket closed:', event.code, event.reason)
        setIsConnected(false)
        wsRef.current = null

        // Attempt to reconnect with exponential backoff
        if (autoConnect && reconnectAttemptsRef.current < maxReconnectAttempts) {
          const delay = Math.min(1000 * Math.pow(2, reconnectAttemptsRef.current), 30000)
          console.log(`⏳ Reconnecting in ${delay}ms (attempt ${reconnectAttemptsRef.current + 1}/${maxReconnectAttempts})`)
          
          reconnectTimeoutRef.current = setTimeout(() => {
            reconnectAttemptsRef.current++
            connect()
          }, delay)
        }
      }

      wsRef.current = ws
    } catch (err) {
      console.error('Failed to create WebSocket:', err)
      setError(err instanceof Error ? err.message : 'Unknown error')
    }
  }, [workflowId, executionId, onMessage, autoConnect])

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
    }
    
    if (wsRef.current) {
      console.log('🔌 Manually disconnecting WebSocket')
      wsRef.current.close()
      wsRef.current = null
    }
    
    setIsConnected(false)
  }, [])

  const send = useCallback((data: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data))
    } else {
      console.warn('WebSocket not connected, cannot send:', data)
    }
  }, [])

  // Auto-connect on mount or when workflowId/executionId changes
  useEffect(() => {
    if (autoConnect && workflowId && executionId) {
      connect()
    }

    return () => {
      disconnect()
    }
  }, [autoConnect, workflowId, executionId, connect, disconnect])

  return {
    isConnected,
    lastMessage,
    error,
    connect,
    disconnect,
    send
  }
}

