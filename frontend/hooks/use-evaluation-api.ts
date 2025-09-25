/**
 * Evaluation API hooks
 * Provides React Query integration for evaluation endpoints
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { toast } from 'react-hot-toast'
import { apiClient } from '@/lib/api-client'

// Query keys for React Query
export const evaluationQueryKeys = {
  history: ['evaluation', 'history'] as const,
  performanceMetrics: ['evaluation', 'performance-metrics'] as const,
  results: (evaluationId: string) => ['evaluation', 'results', evaluationId] as const,
}

// ============= QUERY HOOKS =============

// Get evaluation history
export function useEvaluationHistory() {
  return useQuery({
    queryKey: evaluationQueryKeys.history,
    queryFn: () => apiClient.getEvaluationHistory(),
    refetchInterval: 30000,
    staleTime: 15000,
  })
}

// Get evaluation performance metrics
export function useEvaluationPerformanceMetrics() {
  return useQuery({
    queryKey: evaluationQueryKeys.performanceMetrics,
    queryFn: () => apiClient.getEvaluationPerformanceMetrics(),
    refetchInterval: 30000,
    staleTime: 15000,
  })
}

// Get evaluation results
export function useEvaluationResults(evaluationId: string | null) {
  return useQuery({
    queryKey: evaluationQueryKeys.results(evaluationId!),
    queryFn: () => apiClient.getEvaluationResults(evaluationId!),
    enabled: !!evaluationId,
    refetchInterval: 30000,
  })
}

// ============= MUTATION HOOKS =============

// Evaluate system
export function useEvaluateSystem() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (data: any) => apiClient.evaluateSystem(data),
    onSuccess: (data: any) => {
      queryClient.invalidateQueries({ queryKey: ['evaluation'] })
      toast.success('System evaluation completed!')
      return data
    },
    onError: (error: any) => {
      toast.error(`System evaluation failed: ${error.message}`)
      throw error
    },
  })
}

// Evaluate system quality
export function useEvaluateSystemQuality() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (data: any) => apiClient.evaluateSystemQuality(data),
    onSuccess: (data: any) => {
      queryClient.invalidateQueries({ queryKey: ['evaluation'] })
      toast.success('System quality evaluation completed!')
      return data
    },
    onError: (error: any) => {
      toast.error(`System quality evaluation failed: ${error.message}`)
      throw error
    },
  })
}

// Evaluate component assessment
export function useEvaluateComponentAssessment() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (data: any) => apiClient.evaluateComponentAssessment(data),
    onSuccess: (data: any) => {
      queryClient.invalidateQueries({ queryKey: ['evaluation'] })
      toast.success('Component assessment completed!')
      return data
    },
    onError: (error: any) => {
      toast.error(`Component assessment failed: ${error.message}`)
      throw error
    },
  })
}

// Evaluate integration analysis
export function useEvaluateIntegrationAnalysis() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (data: any) => apiClient.evaluateIntegrationAnalysis(data),
    onSuccess: (data: any) => {
      queryClient.invalidateQueries({ queryKey: ['evaluation'] })
      toast.success('Integration analysis completed!')
      return data
    },
    onError: (error: any) => {
      toast.error(`Integration analysis failed: ${error.message}`)
      throw error
    },
  })
}

// Evaluate benchmark validation
export function useEvaluateBenchmarkValidation() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (data: any) => apiClient.evaluateBenchmarkValidation(data),
    onSuccess: (data: any) => {
      queryClient.invalidateQueries({ queryKey: ['evaluation'] })
      toast.success('Benchmark validation completed!')
      return data
    },
    onError: (error: any) => {
      toast.error(`Benchmark validation failed: ${error.message}`)
      throw error
    },
  })
}

// Evaluate comprehensive
export function useEvaluateComprehensive() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (data: any) => apiClient.evaluateComprehensive(data),
    onSuccess: (data: any) => {
      queryClient.invalidateQueries({ queryKey: ['evaluation'] })
      toast.success('Comprehensive evaluation completed!')
      return data
    },
    onError: (error: any) => {
      toast.error(`Comprehensive evaluation failed: ${error.message}`)
      throw error
    },
  })
}

// Generate assessment report
export function useGenerateAssessmentReport() {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: (data: any) => apiClient.generateAssessmentReport(data),
    onSuccess: (data: any) => {
      queryClient.invalidateQueries({ queryKey: ['evaluation'] })
      toast.success('Assessment report generated!')
      return data
    },
    onError: (error: any) => {
      toast.error(`Failed to generate assessment report: ${error.message}`)
      throw error
    },
  })
}
