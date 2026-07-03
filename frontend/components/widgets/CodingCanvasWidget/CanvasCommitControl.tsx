'use client'

/**
 * CanvasCommitControl — commit + push the session's work (PRD-170 S5)
 *
 * Branch-per-session: the backend commits on `canvas/<session-id>` and pushes
 * with a GitHub App token (server-side; never exposed). The flow: name the repo
 * path inside the workspace → "Prepare" fetches an EDITABLE generated commit
 * message + the changed paths → the user may rewrite it → "Commit & push" lands
 * the branch. No token material ever reaches this component.
 */

import { useCallback, useState } from 'react'
import { GitCommit, Loader2, UploadCloud } from 'lucide-react'

import { apiClient } from '@/lib/api-client'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'

interface CanvasCommitControlProps {
  workspaceId: string | undefined
  /** Bumped by the session on each file edit — re-enables Prepare after work lands. */
  changeSignal: number
}

interface CommitPreview {
  branch: string
  changed_paths: string[]
  message: string
  has_changes: boolean
}

interface CommitResult {
  success: boolean
  branch?: string
  failed_operation?: string
}

export function CanvasCommitControl({ workspaceId, changeSignal }: CanvasCommitControlProps) {
  const [repoPath, setRepoPath] = useState('repos')
  const [preview, setPreview] = useState<CommitPreview | null>(null)
  const [message, setMessage] = useState('')
  const [busy, setBusy] = useState<'idle' | 'preview' | 'commit'>('idle')
  const [error, setError] = useState<string | null>(null)
  const [pushed, setPushed] = useState<string | null>(null)

  const prepare = useCallback(async () => {
    if (!workspaceId || !repoPath.trim()) return
    setBusy('preview')
    setError(null)
    setPushed(null)
    try {
      const res = await apiClient.request<CommitPreview>(
        `/api/workspaces/${workspaceId}/canvas/commit-preview?cwd=${encodeURIComponent(repoPath.trim())}`
      )
      setPreview(res)
      setMessage(res.message)
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to prepare commit')
    } finally {
      setBusy('idle')
    }
  }, [workspaceId, repoPath])

  const commit = useCallback(async () => {
    if (!workspaceId || !repoPath.trim() || !message.trim()) return
    setBusy('commit')
    setError(null)
    try {
      const res = await apiClient.post<CommitResult>(
        `/api/workspaces/${workspaceId}/canvas/commit`,
        { message: message.trim(), cwd: repoPath.trim() }
      )
      if (res.success) {
        setPushed(res.branch ?? 'branch pushed')
        setPreview(null)
      } else {
        setError(`Commit failed at: ${res.failed_operation ?? 'unknown step'}`)
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to commit')
    } finally {
      setBusy('idle')
    }
  }, [workspaceId, repoPath, message])

  return (
    <div className="space-y-2 border-t border-border p-3" data-testid="canvas-commit">
      <div className="flex items-center gap-2">
        <GitCommit className="h-4 w-4 text-muted-foreground" />
        <span className="text-xs font-medium">Commit &amp; push</span>
      </div>

      <div className="flex items-center gap-2">
        <Input
          value={repoPath}
          onChange={(e) => setRepoPath(e.target.value)}
          placeholder="repo path (e.g. repos/my-app)"
          className="h-8 text-xs"
          data-testid="commit-repo-path"
        />
        <Button
          size="sm"
          variant="outline"
          onClick={() => void prepare()}
          disabled={busy !== 'idle' || !repoPath.trim()}
          data-testid="commit-prepare"
          // Re-run when new edits land so the message reflects them.
          key={changeSignal}
        >
          {busy === 'preview' ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : 'Prepare'}
        </Button>
      </div>

      {preview && (
        <div className="space-y-2" data-testid="commit-preview">
          <p className="text-[11px] text-muted-foreground">
            {preview.has_changes
              ? `${preview.changed_paths.length} changed file(s) → ${preview.branch}`
              : 'No changes to commit.'}
          </p>
          <Textarea
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            rows={4}
            className="text-xs font-mono"
            data-testid="commit-message"
          />
          <Button
            size="sm"
            onClick={() => void commit()}
            disabled={busy !== 'idle' || !preview.has_changes || !message.trim()}
            className="w-full"
            data-testid="commit-push"
          >
            {busy === 'commit' ? (
              <Loader2 className="mr-1 h-3.5 w-3.5 animate-spin" />
            ) : (
              <UploadCloud className="mr-1 h-3.5 w-3.5" />
            )}
            Commit &amp; push
          </Button>
        </div>
      )}

      {pushed && (
        <p className="text-[11px] text-emerald-600" data-testid="commit-pushed">
          Pushed {pushed}. Open a PR from the branch on your Git host.
        </p>
      )}
      {error && (
        <p className="text-[11px] text-destructive" data-testid="commit-error">
          {error}
        </p>
      )}
    </div>
  )
}
