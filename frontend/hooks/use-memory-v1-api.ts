/**
 * React hooks for Memory V1 API - All endpoints verified working ✅
 */

'use client'

import { useState, useCallback } from 'react'
import { apiClient } from "@/lib/api-client"

interface StoreMemoryParams {
  session_id: string
  content: any
  memory_type?: string
  importance?: number
  tags?: string[]
}

interface RetrieveMemoryParams {
  sessionId: string
  query?: string
  maxItems?: number
  includeAugmented?: boolean
}

interface ConsolidateMemoryParams {
  session_id?: string
  memory_level?: string
  strategy?: string
}

export function useMemoryStore() {
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [storedMemory, setStoredMemory] = useState<any>(null)

  const storeMemory = useCallback(async (params: StoreMemoryParams) => {
    setIsLoading(true)
    setError(null)

    try {
      // ✅ Verified working: /api/v1/memory/store
      const response = await apiClient.storeMemory(params)
      setStoredMemory(response)
      return response
    } catch (err: any) {
      const errorMsg = err?.message || 'Failed to store memory'
      setError(errorMsg)
      throw err
    } finally {
      setIsLoading(false)
    }
  }, [])

  return {
    isLoading,
    error,
    storedMemory,
    storeMemory,
  }
}

export function useMemoryRetrieval() {
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [memories, setMemories] = useState<any[]>([])

  const retrieveMemory = useCallback(async (params: RetrieveMemoryParams) => {
    setIsLoading(true)
    setError(null)

    try {
      // ✅ Verified working: /api/v1/memory/retrieve/{sessionId}
      const response = await apiClient.retrieveMemory(
        params.sessionId,
        params.query,
        params.maxItems || 20,
        params.includeAugmented !== false
      )
      setMemories(response?.data || [])
      return response
    } catch (err: any) {
      const errorMsg = err?.message || 'Failed to retrieve memory'
      setError(errorMsg)
      throw err
    } finally {
      setIsLoading(false)
    }
  }, [])

  return {
    isLoading,
    error,
    memories,
    retrieveMemory,
  }
}

export function useMemoryConsolidation() {
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [consolidationResult, setConsolidationResult] = useState<any>(null)

  const consolidateMemory = useCallback(async (params?: ConsolidateMemoryParams) => {
    setIsLoading(true)
    setError(null)

    try {
      // ✅ Verified working: /api/v1/memory/consolidate
      const response = await apiClient.consolidateMemory(params)
      setConsolidationResult(response)
      return response
    } catch (err: any) {
      const errorMsg = err?.message || 'Failed to consolidate memory'
      setError(errorMsg)
      throw err
    } finally {
      setIsLoading(false)
    }
  }, [])

  return {
    isLoading,
    error,
    consolidationResult,
    consolidateMemory,
  }
}

export function useMemoryStats() {
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [stats, setStats] = useState<any>(null)

  const fetchStats = useCallback(async () => {
    setIsLoading(true)
    setError(null)

    try {
      // ✅ Verified working: /api/v1/memory/stats
      const response = await apiClient.getMemoryStats()
      setStats(response)
      return response
    } catch (err: any) {
      const errorMsg = err?.message || 'Failed to fetch memory stats'
      setError(errorMsg)
      throw err
    } finally {
      setIsLoading(false)
    }
  }, [])

  return {
    isLoading,
    error,
    stats,
    fetchStats,
  }
}

/**
 * Combined hook for full memory operations
 */
export function useMemoryAPI() {
  const storeOps = useMemoryStore()
  const retrievalOps = useMemoryRetrieval()
  const consolidationOps = useMemoryConsolidation()
  const statsOps = useMemoryStats()

  return {
    // Store operations
    storeMemory: storeOps.storeMemory,
    storedMemory: storeOps.storedMemory,
    storeLoading: storeOps.isLoading,
    storeError: storeOps.error,

    // Retrieval operations
    retrieveMemory: retrievalOps.retrieveMemory,
    memories: retrievalOps.memories,
    retrievalLoading: retrievalOps.isLoading,
    retrievalError: retrievalOps.error,

    // Consolidation operations
    consolidateMemory: consolidationOps.consolidateMemory,
    consolidationResult: consolidationOps.consolidationResult,
    consolidationLoading: consolidationOps.isLoading,
    consolidationError: consolidationOps.error,

    // Stats operations
    fetchStats: statsOps.fetchStats,
    stats: statsOps.stats,
    statsLoading: statsOps.isLoading,
    statsError: statsOps.error,
  }
}


