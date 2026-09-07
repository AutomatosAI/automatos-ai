'use client'

/**
 * PRD-235 W2 — the folders a Code Canvas can be rooted at: the workspace root,
 * every repository under projects/ (LOCAL_PROJECTS_DIR, local edition) and the
 * Claude Code session folders under sessions/. Read through the same files API
 * the explorer uses; a missing folder is simply an empty group.
 */

import { useQuery } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'
import { WORKSPACE_ROOT } from './code-root'

export interface RepoRootOption {
  value: string
  label: string
  group: 'workspace' | 'projects' | 'sessions'
}

interface DirListResponse {
  entries: Array<{ name: string; path: string; type: string }>
}

async function listDirs(workspaceId: string, path: string): Promise<string[]> {
  try {
    const res = await apiClient.request<DirListResponse>(`/api/workspaces/${workspaceId}/files?path=${encodeURIComponent(path)}`)
    return (res.entries || []).filter((e) => e.type === 'directory').map((e) => e.path)
  } catch {
    return []
  }
}

export function repoRootOptions(projects: string[], sessions: string[]): RepoRootOption[] {
  const sessionSorted = [...sessions].sort((a, b) => {
    const na = Number(a.split('/').pop()), nb = Number(b.split('/').pop())
    return Number.isNaN(nb) || Number.isNaN(na) ? b.localeCompare(a) : nb - na
  }).slice(0, 25)
  return [
    { value: WORKSPACE_ROOT, label: 'Workspace root', group: 'workspace' },
    ...projects.map((p) => ({ value: p, label: p.replace(/^projects\//, ''), group: 'projects' as const })),
    ...sessionSorted.map((s) => ({ value: s, label: `ticket ${s.split('/').pop()}`, group: 'sessions' as const })),
  ]
}

export function useRepoRoots(workspaceId: string | undefined) {
  return useQuery({
    queryKey: ['code-roots', workspaceId],
    enabled: !!workspaceId,
    staleTime: 15_000,
    queryFn: async () => {
      const [projects, sessions] = await Promise.all([
        listDirs(workspaceId as string, 'projects'),
        listDirs(workspaceId as string, 'sessions'),
      ])
      return repoRootOptions(projects, sessions)
    },
  })
}
