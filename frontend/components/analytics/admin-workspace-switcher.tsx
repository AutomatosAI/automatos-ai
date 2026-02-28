'use client'

import { useEffect, useCallback } from 'react'
import { Building } from 'lucide-react'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { useQueryClient } from '@tanstack/react-query'
import { setAdminWorkspaceOverride, getAdminWorkspaceOverride } from '@/lib/api-client'
import { useAdminWorkspaceAnalytics } from '@/hooks/use-unified-analytics'

interface Props {
  onWorkspaceChange?: (label: string) => void
}

export function AdminWorkspaceSwitcher({ onWorkspaceChange }: Props) {
  const queryClient = useQueryClient()
  const { data } = useAdminWorkspaceAnalytics(30)
  const current = getAdminWorkspaceOverride() || 'mine'

  const workspaces = data?.workspaces ?? []

  const handleChange = useCallback((value: string) => {
    if (value === 'mine') {
      setAdminWorkspaceOverride(null)
      onWorkspaceChange?.('My Workspace')
    } else if (value === '__all__') {
      setAdminWorkspaceOverride('__all__')
      onWorkspaceChange?.('All Workspaces')
    } else {
      setAdminWorkspaceOverride(value)
      const ws = workspaces.find((w: any) => w.id === value)
      onWorkspaceChange?.(ws?.name || value)
    }
    // Invalidate all analytics queries to refetch with new workspace context
    queryClient.invalidateQueries({ queryKey: ['unified-analytics'] })
  }, [queryClient, workspaces, onWorkspaceChange])

  // Reset override on unmount (leaving analytics page)
  useEffect(() => {
    return () => {
      setAdminWorkspaceOverride(null)
    }
  }, [])

  return (
    <Select value={current} onValueChange={handleChange}>
      <SelectTrigger className="w-44 bg-secondary/50">
        <Building className="w-3.5 h-3.5 mr-1.5 text-muted-foreground" />
        <SelectValue placeholder="My Workspace" />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value="mine">My Workspace</SelectItem>
        <SelectItem value="__all__">All Workspaces</SelectItem>
        {workspaces
          .filter((ws: any) => ws.id !== 'current')
          .map((ws: any) => (
            <SelectItem key={ws.id} value={ws.id}>
              {ws.name}
            </SelectItem>
          ))}
      </SelectContent>
    </Select>
  )
}
