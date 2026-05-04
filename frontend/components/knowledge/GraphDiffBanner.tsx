'use client'

import { useQuery } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'
import { useWorkspace } from '@/hooks/use-workspace'
import { AlertCircle, ArrowRight } from 'lucide-react'

interface GraphDiff {
  new_nodes: string[]
  removed_nodes: string[]
  new_edges: Array<unknown>
  removed_edges: Array<unknown>
  summary?: string
  build_report_path?: string
  built_at?: string
}

export function GraphDiffBanner() {
  const { workspaceId } = useWorkspace()

  const { data: diff } = useQuery<GraphDiff | null>({
    queryKey: ['business-graph-diff', workspaceId],
    queryFn: async () => {
      if (!workspaceId) return null
      try {
        const content = await apiClient.getWorkspaceFileContent(
          workspaceId,
          '/graph/latest_diff.json'
        )
        if (!content) return null
        // content may already be parsed or may be a string
        const parsed = typeof content === 'string' ? JSON.parse(content) : content
        return parsed as GraphDiff
      } catch {
        // File doesn't exist or isn't readable — not an error
        return null
      }
    },
    enabled: !!workspaceId,
    staleTime: 60_000,
    retry: false,
  })

  if (!diff) return null

  const newCount = diff.new_nodes?.length ?? 0
  const removedCount = diff.removed_nodes?.length ?? 0
  const hasChanges = newCount > 0 || removedCount > 0

  if (!hasChanges) return null

  const parts: string[] = []
  if (newCount > 0) parts.push(`${newCount} new node${newCount !== 1 ? 's' : ''}`)
  if (removedCount > 0) parts.push(`${removedCount} removed`)

  return (
    <div className="bg-info/10 border border-info/20 rounded-lg p-3 flex items-center justify-between">
      <div className="flex items-center gap-2 text-sm">
        <AlertCircle className="w-4 h-4 text-info shrink-0" />
        <span className="text-info/80">
          {parts.join(', ')} since last build
        </span>
        {diff.built_at && (
          <span className="text-info/60 text-xs">
            ({new Date(diff.built_at).toLocaleDateString()})
          </span>
        )}
      </div>
      {diff.build_report_path && (
        <button
          className="text-xs text-info hover:text-info/80 flex items-center gap-1 transition-colors"
          onClick={() => {
            // Could open report viewer — for now just log
            console.log('View build report:', diff.build_report_path)
          }}
        >
          View report
          <ArrowRight className="w-3 h-3" />
        </button>
      )}
    </div>
  )
}
