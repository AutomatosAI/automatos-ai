'use client'

/**
 * RepoSelector — Dialog for listing and cloning GitHub repos
 * PRD-66 Phase 2: GitHub Repo Integration
 */

import { useCallback, useEffect, useState } from 'react'
import { GitBranch, Loader2, Search, Lock, Globe, AlertCircle } from 'lucide-react'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { cn } from '@/lib/utils'
import { apiClient } from '@/lib/api-client'

interface GithubRepo {
  name: string
  full_name: string
  url: string
  description: string
  default_branch: string
  private: boolean
  language: string | null
  updated_at: string | null
}

interface RepoSelectorProps {
  workspaceId: string
  open: boolean
  onOpenChange: (open: boolean) => void
  onCloneStarted?: (taskId: string) => void
}

export function RepoSelector({
  workspaceId,
  open,
  onOpenChange,
  onCloneStarted,
}: RepoSelectorProps) {
  const [repos, setRepos] = useState<GithubRepo[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [filter, setFilter] = useState('')
  const [cloningRepo, setCloningRepo] = useState<string | null>(null)

  // Fetch repos when dialog opens
  useEffect(() => {
    if (!open) return
    setIsLoading(true)
    setError(null)

    apiClient
      .listGithubRepos(workspaceId, 1, 100)
      .then((data) => {
        const typed = data as { repos: GithubRepo[] }
        setRepos(typed.repos || [])
      })
      .catch((err: Error) => {
        setError(err.message || 'Failed to load repos. Is GitHub connected via Composio?')
      })
      .finally(() => setIsLoading(false))
  }, [open, workspaceId])

  const handleClone = useCallback(
    async (repo: GithubRepo) => {
      setCloningRepo(repo.full_name)
      try {
        const result = (await apiClient.cloneGithubRepo(
          workspaceId,
          repo.url,
          repo.default_branch
        )) as { task_id: string }
        onCloneStarted?.(result.task_id)
        onOpenChange(false)
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : 'Clone failed'
        setError(msg)
      } finally {
        setCloningRepo(null)
      }
    },
    [workspaceId, onCloneStarted, onOpenChange]
  )

  const filtered = repos.filter(
    (r) =>
      r.full_name.toLowerCase().includes(filter.toLowerCase()) ||
      (r.description || '').toLowerCase().includes(filter.toLowerCase())
  )

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-lg max-h-[80vh] flex flex-col">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <GitBranch className="h-4 w-4" />
            Clone GitHub Repository
          </DialogTitle>
          <DialogDescription>
            Select a repo to clone into your workspace.
          </DialogDescription>
        </DialogHeader>

        {/* Search */}
        <div className="relative">
          <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Filter repos..."
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            className="pl-9 h-9 text-sm"
          />
        </div>

        {/* Repo list */}
        <div className="flex-1 overflow-y-auto min-h-[200px] max-h-[400px] border rounded-md">
          {isLoading && (
            <div className="flex items-center justify-center h-32">
              <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
              <span className="ml-2 text-sm text-muted-foreground">Loading repos...</span>
            </div>
          )}

          {error && (
            <div className="flex items-center gap-2 p-4 text-sm text-destructive">
              <AlertCircle className="h-4 w-4 flex-shrink-0" />
              {error}
            </div>
          )}

          {!isLoading && !error && filtered.length === 0 && (
            <div className="p-4 text-sm text-muted-foreground text-center">
              {repos.length === 0
                ? 'No repos found. Connect GitHub via Settings > Composio first.'
                : 'No repos match your filter.'}
            </div>
          )}

          {!isLoading &&
            filtered.map((repo) => (
              <button
                key={repo.full_name}
                onClick={() => handleClone(repo)}
                disabled={cloningRepo !== null}
                className={cn(
                  'w-full text-left px-3 py-2.5 border-b last:border-b-0',
                  'hover:bg-muted/50 transition-colors',
                  'disabled:opacity-50 disabled:cursor-not-allowed'
                )}
              >
                <div className="flex items-center gap-2">
                  {repo.private ? (
                    <Lock className="h-3.5 w-3.5 text-warning flex-shrink-0" />
                  ) : (
                    <Globe className="h-3.5 w-3.5 text-muted-foreground flex-shrink-0" />
                  )}
                  <span className="font-medium text-sm truncate">{repo.full_name}</span>
                  {repo.language && (
                    <span className="ml-auto text-[10px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground flex-shrink-0">
                      {repo.language}
                    </span>
                  )}
                  {cloningRepo === repo.full_name && (
                    <Loader2 className="h-3.5 w-3.5 animate-spin ml-auto flex-shrink-0" />
                  )}
                </div>
                {repo.description && (
                  <p className="text-xs text-muted-foreground mt-0.5 truncate">{repo.description}</p>
                )}
              </button>
            ))}
        </div>
      </DialogContent>
    </Dialog>
  )
}
