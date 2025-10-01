/**
 * Usage Analytics API Hook
 */

import { useQuery } from '@tanstack/react-query'

export interface UsageAnalyticsData {
  period: string
  start_time: string
  end_time: string
  total_events: number
  event_counts: {
    document_viewed: number
    document_searched: number
    chunk_retrieved: number
    rag_query: number
  }
  popular_documents: Array<{
    document_id: number
    filename: string
    file_type: string
    view_count: number
  }>
  popular_search_terms: Array<{
    query: string
    count: number
  }>
  time_series: Array<{
    date: string
    count: number
  }>
  summary: {
    total_documents: number
    processed_documents: number
    documents_this_period: number
  }
}

export function useUsageAnalytics(period: string = '7d') {
  return useQuery({
    queryKey: ['documents', 'analytics', 'usage', period],
    queryFn: async () => {
      const response = await fetch(
        `/api/documents/analytics/usage?period=${period}`,
        {
          headers: { 'Content-Type': 'application/json' },
        }
      )

      if (!response.ok) {
        throw new Error(`Failed to fetch usage analytics: ${response.statusText}`)
      }

      return response.json() as Promise<UsageAnalyticsData>
    },
    staleTime: 60000, // Cache for 1 minute
    refetchInterval: 300000, // Refetch every 5 minutes
  })
}


