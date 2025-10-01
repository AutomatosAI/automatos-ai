
/**
 * React hooks for Field Theory Integration
 */

'use client'

import { useState, useCallback, useEffect } from 'react'
import { apiClient } from "@/lib/api-client"

export function useFieldOperations() {
  const [isLoading, setIsLoading] = useState(false)
  const [fieldData, setFieldData] = useState<any>(null)
  const [error, setError] = useState<string | null>(null)

  const updateField = useCallback(async (
    sessionId: string,
    contextData: any,
    fieldType?: 'scalar' | 'vector' | 'tensor'
  ) => {
    setIsLoading(true)
    setError(null)

    try {
      // ✅ Using verified working endpoint: /api/field-theory/fields/update
      const response = await apiClient.updateFieldContext({
        session_id: sessionId,
        context_data: contextData,
        field_type: fieldType,
      })

      setFieldData(response)
      setIsLoading(false)
      return response
    } catch (err: any) {
      setError(err?.message || 'Failed to update field')
      setIsLoading(false)
      throw err
    }
  }, [])

  const propagateField = useCallback(async (
    source: string,
    targets?: string[],
    propagationSteps?: number
  ) => {
    setIsLoading(true)
    setError(null)

    try {
      // ✅ Using verified working endpoint: /api/field-theory/fields/propagate
      const response = await apiClient.propagateField({
        source,
        targets,
        propagation_steps: propagationSteps,
      })

      setFieldData(response)
      setIsLoading(false)
      return response
    } catch (err: any) {
      setError(err?.message || 'Failed to propagate field')
      setIsLoading(false)
      throw err
    }
  }, [])

  // Note: optimizeField endpoint not yet working - remove or implement when backend ready
  // const optimizeField = ... removed until backend implements /api/field-theory/fields/optimize

  return {
    isLoading,
    fieldData,
    error,
    updateField,
    propagateField,
    // optimizeField - removed until backend ready
  }
}

export function useFieldInteractions() {
  const [isLoading, setIsLoading] = useState(false)
  const [interactionData, setInteractionData] = useState<any>(null)
  const [error, setError] = useState<string | null>(null)

  const modelInteractions = useCallback(async (
    taskId: number,
    userId: number,
    similarityThreshold?: number
  ) => {
    setIsLoading(true)
    setError(null)

    try {
      // ✅ Using verified working endpoint: /api/field-theory/fields/interactions
      const response = await apiClient.modelFieldInteractions({
        task_id: taskId,
        user_id: userId,
        similarity_threshold: similarityThreshold,
      })

      setInteractionData(response)
      setIsLoading(false)
      return response
    } catch (err: any) {
      setError(err?.message || 'Failed to model field interactions')
      setIsLoading(false)
      throw err
    }
  }, [])

  const dynamicManagement = useCallback(async (
    sessionId: string,
    context?: any
  ) => {
    setIsLoading(true)
    setError(null)

    try {
      // ✅ Using verified working endpoint: /api/field-theory/fields/dynamic
      const response = await apiClient.manageDynamicFields({
        session_id: sessionId,
        context,
      })

      setInteractionData(response)
      setIsLoading(false)
      return response
    } catch (err: any) {
      setError(err?.message || 'Failed to manage dynamic fields')
      setIsLoading(false)
      throw err
    }
  }, [])

  return {
    isLoading,
    interactionData,
    error,
    modelInteractions,
    dynamicManagement,
  }
}

// useFieldContext removed - endpoints not yet implemented
// Use updateFieldContext and manageDynamicFields from useFieldOperations instead

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
      // ✅ Using verified working endpoint: /api/field-theory/health
      const health = await apiClient.getFieldTheoryHealth()

      setStatistics({
        health,
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

// Batch operations removed - not yet implemented in backend
// export function useBatchFieldOperations() { ... }
