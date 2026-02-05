/**
 * Processing Queue WebSocket Hook
 * 
 * Provides real-time updates for document processing status
 * 
 * MVP Implementation: Uses smart polling (refetch every 2 seconds)
 * Future Enhancement: Can be upgraded to true WebSocket connection
 */

import { useEffect, useState } from 'react'
import { useProcessingQueue, type QueueStatus } from './use-processing-queue-api'

export interface WebSocketStatus {
  isConnected: boolean
  lastUpdate: string | null
  error: string | null
}

/**
 * Hook that provides "live" queue updates via smart polling
 * 
 * This is a pragmatic MVP approach that:
 * - Works immediately without WebSocket infrastructure
 * - Provides near-real-time updates (2s interval)
 * - Can be upgraded to true WebSocket later
 * - Handles connection state
 */
export function useProcessingQueueWebSocket() {
  const [wsStatus, setWsStatus] = useState<WebSocketStatus>({
    isConnected: false,
    lastUpdate: null,
    error: null,
  })

  // Use React Query with smart polling (balanced between real-time and API costs)
  const {
    data: queueStatus,
    isLoading,
    error,
    isSuccess,
  } = useProcessingQueue({
    enabled: true,
    refetchInterval: 10000, // Poll every 10 seconds (reduced from 2s to save API costs)
  })

  // Update connection status based on query state
  useEffect(() => {
    if (isSuccess) {
      setWsStatus({
        isConnected: true,
        lastUpdate: queueStatus?.timestamp || null,
        error: null,
      })
    } else if (error) {
      setWsStatus({
        isConnected: false,
        lastUpdate: null,
        error: (error as Error).message,
      })
    }
  }, [isSuccess, error, queueStatus])

  return {
    queueStatus: queueStatus || null,
    isConnected: wsStatus.isConnected,
    lastUpdate: wsStatus.lastUpdate,
    error: wsStatus.error,
    isLoading,
  }
}

/**
 * Future WebSocket Implementation (commented for reference)
 * 
 * Uncomment and use when backend WebSocket endpoint is ready:
 * 
 * export function useProcessingQueueWebSocketReal() {
 *   const [queueStatus, setQueueStatus] = useState<QueueStatus | null>(null)
 *   const [isConnected, setIsConnected] = useState(false)
 *   
 *   useEffect(() => {
 *     const ws = new WebSocket(`wss://${process.env.NEXT_PUBLIC_API_URL?.replace('https://', '') || 'your-api-url.com'}/ws/documents/processing`)
 *     
 *     ws.onopen = () => {
 *       setIsConnected(true)
 *       console.log('WebSocket connected')
 *     }
 *     
 *     ws.onmessage = (event) => {
 *       const update = JSON.parse(event.data)
 *       setQueueStatus(update)
 *     }
 *     
 *     ws.onerror = (error) => {
 *       console.error('WebSocket error:', error)
 *       setIsConnected(false)
 *     }
 *     
 *     ws.onclose = () => {
 *       console.log('WebSocket closed')
 *       setIsConnected(false)
 *     }
 *     
 *     return () => {
 *       ws.close()
 *     }
 *   }, [])
 *   
 *   return { queueStatus, isConnected }
 * }
 */


