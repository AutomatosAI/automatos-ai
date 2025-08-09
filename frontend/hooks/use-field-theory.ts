
/**
 * React hooks for Field Theory Integration
 */

'use client'

import { useState, useCallback, useEffect } from 'react'
import { fieldTheoryClient } from '@/lib/api/field-theory-client'

export function useFieldOperations() {
  const [isLoading, setIsLoading] = useState(false)
  const [fieldData, setFieldData] = useState<any>(null)
  const [error, setError] = useState<string | null>(null)

  const updateField = useCallback(async (
    sessionId: string,
    context: string,
    fieldType?: 'scalar' | 'vector' | 'tensor',
    influenceWeights?: number[]
  ) => {
    setIsLoading(true)
    setError(null)

    const response = await fieldTheoryClient.updateField({
      session_id: sessionId,
      context,
      field_type: fieldType,
      influence_weights: influenceWeights,
    })

    if (response.error) {
      setError(response.error)
    } else {
      setFieldData(response.data)
    }

    setIsLoading(false)
    return response
  }, [])

  const propagateField = useCallback(async (
    sessionId: string,
    steps?: number,
    alpha?: number,
    beta?: number
  ) => {
    setIsLoading(true)
    setError(null)

    const response = await fieldTheoryClient.propagateField({
      session_id: sessionId,
      steps,
      alpha,
      beta,
    })

    if (response.error) {
      setError(response.error)
    } else {
      setFieldData(response.data)
    }

    setIsLoading(false)
    return response
  }, [])

  const optimizeField = useCallback(async (
    sessionId: string,
    objectives?: string[],
    constraints?: Record<string, any>
  ) => {
    setIsLoading(true)
    setError(null)

    const response = await fieldTheoryClient.optimizeField({
      session_id: sessionId,
      objectives,
      constraints,
    })

    if (response.error) {
      setError(response.error)
    } else {
      setFieldData(response.data)
    }

    setIsLoading(false)
    return response
  }, [])

  return {
    isLoading,
    fieldData,
    error,
    updateField,
    propagateField,
    optimizeField,
  }
}

export function useFieldInteractions() {
  const [isLoading, setIsLoading] = useState(false)
  const [interactionData, setInteractionData] = useState<any>(null)
  const [error, setError] = useState<string | null>(null)

  const modelInteractions = useCallback(async (
    sessionId1: string,
    sessionId2: string,
    interactionType?: 'similarity' | 'difference' | 'combination'
  ) => {
    setIsLoading(true)
    setError(null)

    const response = await fieldTheoryClient.modelFieldInteractions({
      session_id_1: sessionId1,
      session_id_2: sessionId2,
      interaction_type: interactionType,
    })

    if (response.error) {
      setError(response.error)
    } else {
      setInteractionData(response.data)
    }

    setIsLoading(false)
    return response
  }, [])

  const dynamicManagement = useCallback(async (
    sessionId: string,
    timeDelta?: number,
    learningRate?: number,
    momentum?: number
  ) => {
    setIsLoading(true)
    setError(null)

    const response = await fieldTheoryClient.dynamicFieldManagement({
      session_id: sessionId,
      time_delta: timeDelta,
      learning_rate: learningRate,
      momentum,
    })

    if (response.error) {
      setError(response.error)
    } else {
      setInteractionData(response.data)
    }

    setIsLoading(false)
    return response
  }, [])

  return {
    isLoading,
    interactionData,
    error,
    modelInteractions,
    dynamicManagement,
  }
}

export function useFieldContext() {
  const [contexts, setContexts] = useState<Record<string, any>>({})
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const getContext = useCallback(async (sessionId: string) => {
    setIsLoading(true)
    setError(null)

    const response = await fieldTheoryClient.getFieldContext(sessionId)

    if (response.error) {
      setError(response.error)
    } else if (response.data) {
      setContexts(prev => ({ ...prev, [sessionId]: response.data }))
    }

    setIsLoading(false)
    return response
  }, [])

  const clearContext = useCallback(async (sessionId: string) => {
    setIsLoading(true)
    setError(null)

    const response = await fieldTheoryClient.clearFieldContext(sessionId)

    if (response.error) {
      setError(response.error)
    } else {
      setContexts(prev => {
        const newContexts = { ...prev }
        delete newContexts[sessionId]
        return newContexts
      })
    }

    setIsLoading(false)
    return response
  }, [])

  return {
    contexts,
    isLoading,
    error,
    getContext,
    clearContext,
  }
}

export function useFieldStatistics() {
  const [statistics, setStatistics] = useState<{
    fieldStats?: any
    fieldStates?: any
    interactions?: any
    health?: any
  }>({})
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchAllStatistics = useCallback(async () => {
    setIsLoading(true)
    setError(null)

    try {
      const [fieldStats, fieldStates, interactions, health] = await Promise.all([
        fieldTheoryClient.getFieldStatistics(),
        fieldTheoryClient.getFieldStates(),
        fieldTheoryClient.getFieldInteractions(),
        fieldTheoryClient.healthCheck(),
      ])

      setStatistics({
        fieldStats: fieldStats.data,
        fieldStates: fieldStates.data,
        interactions: interactions.data,
        health: health.data,
      })
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch statistics')
    }

    setIsLoading(false)
  }, [])

  useEffect(() => {
    fetchAllStatistics()
  }, [fetchAllStatistics])

  return {
    statistics,
    isLoading,
    error,
    refetch: fetchAllStatistics,
  }
}

export function useBatchFieldOperations() {
  const [isLoading, setIsLoading] = useState(false)
  const [batchResults, setBatchResults] = useState<any[]>([])
  const [error, setError] = useState<string | null>(null)

  const batchUpdateFields = useCallback(async (requests: Array<{
    session_id: string
    context: string
    field_type?: 'scalar' | 'vector' | 'tensor'
    influence_weights?: number[]
  }>) => {
    setIsLoading(true)
    setError(null)

    const response = await fieldTheoryClient.batchUpdateFields(requests)

    if (response.error) {
      setError(response.error)
    } else {
      setBatchResults(response.data || [])
    }

    setIsLoading(false)
    return response
  }, [])

  const batchPropagateFields = useCallback(async (requests: Array<{
    session_id: string
    steps?: number
    alpha?: number
    beta?: number
  }>) => {
    setIsLoading(true)
    setError(null)

    const response = await fieldTheoryClient.batchPropagateFields(requests)

    if (response.error) {
      setError(response.error)
    } else {
      setBatchResults(response.data || [])
    }

    setIsLoading(false)
    return response
  }, [])

  return {
    isLoading,
    batchResults,
    error,
    batchUpdateFields,
    batchPropagateFields,
  }
}
