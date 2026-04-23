'use client'

import { useState, useEffect, useCallback } from 'react'
import {
  Building2,
  Users,
  FileText,
  HardDrive,
  MessageSquare,
  Pause,
  Play,
  Trash2,
  RotateCcw,
  Copy,
  Check,
  Search,
  Loader2,
  AlertTriangle,
  RefreshCw,
} from 'lucide-react'
import { MainLayout } from '@/components/layout/main-layout'
import { Card, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { usePageAPI } from '@/hooks/use-page-api'
import { apiClient } from '@/lib/api-client'

// ===================================================================
// Types
// ===================================================================

interface WorkspaceRow {
  id: string
  name: string
  slug: string | null
  plan: string | null
  is_personal: boolean
  is_active: boolean
  owner_email: string | null
  owner_name: string | null
  agents_count: number
  documents_count: number
  storage_bytes: number
  chats_count: number
  created_at: string | null
  paused_at: string | null
  paused_reason: string | null
  deleted_at: string | null
}

interface ListResponse {
  items: WorkspaceRow[]
  total: number
  page: number
  limit: number
}

// ===================================================================
// Helpers
// ===================================================================

function formatBytes(bytes: number): string {
  if (!bytes) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let i = 0
  let n = bytes
  while (n >= 1024 && i < units.length - 1) {
    n /= 1024
    i++
  }
  return `${n.toFixed(i === 0 ? 0 : 1)} ${units[i]}`
}

function formatDate(iso: string | null): string {
  if (!iso) return '—'
  return new Date(iso).toLocaleDateString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  })
}

function stateBadge(w: WorkspaceRow) {
  if (w.deleted_at) {
    return (
      <Badge className="bg-red-500/20 text-red-400 border-red-500/30">
        Deleted
      </Badge>
    )
  }
  if (w.paused_at) {
    return (
      <Badge className="bg-yellow-500/20 text-yellow-400 border-yellow-500/30">
        Paused
      </Badge>
    )
  }
  return (
    <Badge className="bg-green-500/20 text-green-400 border-green-500/30">
      Active
    </Badge>
  )
}

// ===================================================================
// Page
// ===================================================================

export default function AdminWorkspacesPage() {
  usePageAPI('admin')

  const [workspaces, setWorkspaces] = useState<WorkspaceRow[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [limit] = useState(50)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [search, setSearch] = useState('')
  const [includeDeleted, setIncludeDeleted] = useState(false)
  const [sort, setSort] = useState<'created_at' | 'name' | 'storage_bytes' | 'agents_count'>('created_at')
  const [order, setOrder] = useState<'asc' | 'desc'>('desc')

  const [actionId, setActionId] = useState<string | null>(null)
  const [copiedId, setCopiedId] = useState<string | null>(null)

  // Pause modal state
  const [pauseTarget, setPauseTarget] = useState<WorkspaceRow | null>(null)
  const [pauseReason, setPauseReason] = useState('')

  // Delete confirmation
  const [deleteTarget, setDeleteTarget] = useState<WorkspaceRow | null>(null)
  const [deleteConfirm, setDeleteConfirm] = useState('')

  const fetchWorkspaces = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const params = new URLSearchParams({
        page: String(page),
        limit: String(limit),
        sort,
        order,
        include_deleted: String(includeDeleted),
      })
      if (search.trim()) params.set('search', search.trim())

      const data = await apiClient.request<ListResponse>(
        `/api/admin/workspaces?${params.toString()}`
      )
      setWorkspaces(data.items)
      setTotal(data.total)
    } catch (err: any) {
      setError(err?.message || 'Failed to load workspaces')
    } finally {
      setLoading(false)
    }
  }, [page, limit, sort, order, includeDeleted, search])

  useEffect(() => {
    fetchWorkspaces()
  }, [fetchWorkspaces])

  async function copyWorkspaceId(id: string) {
    try {
      await navigator.clipboard.writeText(id)
      setCopiedId(id)
      setTimeout(() => setCopiedId(null), 1500)
    } catch {
      /* noop */
    }
  }

  async function handlePause() {
    if (!pauseTarget || !pauseReason.trim()) return
    setActionId(pauseTarget.id)
    try {
      await apiClient.request(`/api/admin/workspaces/${pauseTarget.id}/pause`, {
        method: 'POST',
        body: { reason: pauseReason.trim() },
      })
      setPauseTarget(null)
      setPauseReason('')
      await fetchWorkspaces()
    } catch (err: any) {
      setError(err?.message || 'Failed to pause workspace')
    } finally {
      setActionId(null)
    }
  }

  async function handleResume(id: string) {
    setActionId(id)
    try {
      await apiClient.request(`/api/admin/workspaces/${id}/resume`, {
        method: 'POST',
      })
      await fetchWorkspaces()
    } catch (err: any) {
      setError(err?.message || 'Failed to resume workspace')
    } finally {
      setActionId(null)
    }
  }

  async function handleDelete() {
    if (!deleteTarget || deleteConfirm !== deleteTarget.name) return
    setActionId(deleteTarget.id)
    try {
      await apiClient.request(`/api/admin/workspaces/${deleteTarget.id}`, {
        method: 'DELETE',
      })
      setDeleteTarget(null)
      setDeleteConfirm('')
      await fetchWorkspaces()
    } catch (err: any) {
      setError(err?.message || 'Failed to delete workspace')
    } finally {
      setActionId(null)
    }
  }

  async function handleRestore(id: string) {
    setActionId(id)
    try {
      await apiClient.request(`/api/admin/workspaces/${id}/restore`, {
        method: 'POST',
      })
      await fetchWorkspaces()
    } catch (err: any) {
      setError(err?.message || 'Failed to restore workspace')
    } finally {
      setActionId(null)
    }
  }

  const totalStorage = workspaces.reduce((sum, w) => sum + w.storage_bytes, 0)
  const totalAgents = workspaces.reduce((sum, w) => sum + w.agents_count, 0)
  const pausedCount = workspaces.filter((w) => w.paused_at && !w.deleted_at).length
  const deletedCount = workspaces.filter((w) => w.deleted_at).length

  return (
    <MainLayout>
      <div className="p-6 space-y-6 max-w-[1600px] mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold flex items-center gap-2">
              <Building2 className="h-6 w-6" />
              Workspace Admin
            </h1>
            <p className="text-sm text-muted-foreground mt-1">
              Manage all workspaces — pause, delete, inspect usage.
            </p>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={fetchWorkspaces}
            disabled={loading}
          >
            <RefreshCw className={`h-4 w-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card>
            <CardContent className="p-4">
              <div className="text-xs text-muted-foreground">Total workspaces</div>
              <div className="text-2xl font-bold mt-1">{total}</div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4">
              <div className="text-xs text-muted-foreground">Storage (visible page)</div>
              <div className="text-2xl font-bold mt-1">{formatBytes(totalStorage)}</div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4">
              <div className="text-xs text-muted-foreground">Agents (visible page)</div>
              <div className="text-2xl font-bold mt-1">{totalAgents}</div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4">
              <div className="text-xs text-muted-foreground">Paused / Deleted</div>
              <div className="text-2xl font-bold mt-1">
                {pausedCount} / {deletedCount}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Filters */}
        <Card>
          <CardContent className="p-4 flex flex-wrap items-center gap-3">
            <div className="relative flex-1 min-w-[200px]">
              <Search className="h-4 w-4 absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
              <Input
                placeholder="Search name or slug…"
                value={search}
                onChange={(e) => {
                  setSearch(e.target.value)
                  setPage(1)
                }}
                className="pl-9"
              />
            </div>
            <select
              value={sort}
              onChange={(e) => setSort(e.target.value as any)}
              className="bg-background border rounded-md px-3 py-2 text-sm"
            >
              <option value="created_at">Sort: Created</option>
              <option value="name">Sort: Name</option>
              <option value="storage_bytes">Sort: Storage</option>
              <option value="agents_count">Sort: Agents</option>
            </select>
            <select
              value={order}
              onChange={(e) => setOrder(e.target.value as any)}
              className="bg-background border rounded-md px-3 py-2 text-sm"
            >
              <option value="desc">Desc</option>
              <option value="asc">Asc</option>
            </select>
            <label className="flex items-center gap-2 text-sm cursor-pointer">
              <input
                type="checkbox"
                checked={includeDeleted}
                onChange={(e) => {
                  setIncludeDeleted(e.target.checked)
                  setPage(1)
                }}
              />
              Include deleted
            </label>
          </CardContent>
        </Card>

        {error && (
          <Card className="border-red-500/30 bg-red-500/5">
            <CardContent className="p-4 flex items-center gap-2 text-red-400">
              <AlertTriangle className="h-4 w-4" />
              {error}
            </CardContent>
          </Card>
        )}

        {/* Table */}
        <Card>
          <CardContent className="p-0">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-muted/40 text-xs uppercase text-muted-foreground">
                  <tr>
                    <th className="text-left px-4 py-3">Workspace</th>
                    <th className="text-left px-4 py-3">Owner</th>
                    <th className="text-left px-4 py-3">Plan</th>
                    <th className="text-left px-4 py-3">State</th>
                    <th className="text-right px-4 py-3">
                      <Users className="h-3.5 w-3.5 inline" />
                    </th>
                    <th className="text-right px-4 py-3">
                      <FileText className="h-3.5 w-3.5 inline" />
                    </th>
                    <th className="text-right px-4 py-3">
                      <HardDrive className="h-3.5 w-3.5 inline" />
                    </th>
                    <th className="text-right px-4 py-3">
                      <MessageSquare className="h-3.5 w-3.5 inline" />
                    </th>
                    <th className="text-left px-4 py-3">Created</th>
                    <th className="text-right px-4 py-3">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {loading && workspaces.length === 0 && (
                    <tr>
                      <td colSpan={10} className="px-4 py-12 text-center">
                        <Loader2 className="h-6 w-6 animate-spin mx-auto text-muted-foreground" />
                      </td>
                    </tr>
                  )}
                  {!loading && workspaces.length === 0 && (
                    <tr>
                      <td colSpan={10} className="px-4 py-12 text-center text-muted-foreground">
                        No workspaces found.
                      </td>
                    </tr>
                  )}
                  {workspaces.map((w) => (
                    <tr
                      key={w.id}
                      className="border-t border-border hover:bg-muted/20"
                    >
                      <td className="px-4 py-3">
                        <div className="font-medium">{w.name}</div>
                        <div className="flex items-center gap-1 mt-0.5">
                          <code className="text-[10px] font-mono text-muted-foreground">
                            {w.id}
                          </code>
                          <button
                            onClick={() => copyWorkspaceId(w.id)}
                            className="opacity-60 hover:opacity-100"
                            title="Copy UUID"
                          >
                            {copiedId === w.id ? (
                              <Check className="h-3 w-3 text-green-400" />
                            ) : (
                              <Copy className="h-3 w-3" />
                            )}
                          </button>
                        </div>
                      </td>
                      <td className="px-4 py-3">
                        <div>{w.owner_name || '—'}</div>
                        <div className="text-xs text-muted-foreground">
                          {w.owner_email || '—'}
                        </div>
                      </td>
                      <td className="px-4 py-3">
                        <Badge variant="outline" className="capitalize">
                          {w.plan || '—'}
                        </Badge>
                        {w.is_personal && (
                          <Badge variant="outline" className="ml-1 text-[10px]">
                            personal
                          </Badge>
                        )}
                      </td>
                      <td className="px-4 py-3">{stateBadge(w)}</td>
                      <td className="px-4 py-3 text-right tabular-nums">
                        {w.agents_count}
                      </td>
                      <td className="px-4 py-3 text-right tabular-nums">
                        {w.documents_count}
                      </td>
                      <td className="px-4 py-3 text-right tabular-nums">
                        {formatBytes(w.storage_bytes)}
                      </td>
                      <td className="px-4 py-3 text-right tabular-nums">
                        {w.chats_count}
                      </td>
                      <td className="px-4 py-3 text-xs text-muted-foreground">
                        {formatDate(w.created_at)}
                      </td>
                      <td className="px-4 py-3 text-right">
                        <div className="flex items-center justify-end gap-1">
                          {w.deleted_at ? (
                            <Button
                              size="sm"
                              variant="outline"
                              onClick={() => handleRestore(w.id)}
                              disabled={actionId === w.id}
                              title="Restore"
                            >
                              <RotateCcw className="h-3.5 w-3.5" />
                            </Button>
                          ) : (
                            <>
                              {w.paused_at ? (
                                <Button
                                  size="sm"
                                  variant="outline"
                                  onClick={() => handleResume(w.id)}
                                  disabled={actionId === w.id}
                                  title="Resume"
                                >
                                  <Play className="h-3.5 w-3.5" />
                                </Button>
                              ) : (
                                <Button
                                  size="sm"
                                  variant="outline"
                                  onClick={() => {
                                    setPauseTarget(w)
                                    setPauseReason('')
                                  }}
                                  disabled={actionId === w.id}
                                  title="Pause"
                                >
                                  <Pause className="h-3.5 w-3.5" />
                                </Button>
                              )}
                              <Button
                                size="sm"
                                variant="outline"
                                onClick={() => {
                                  setDeleteTarget(w)
                                  setDeleteConfirm('')
                                }}
                                disabled={actionId === w.id}
                                className="text-red-400 hover:text-red-300"
                                title="Delete"
                              >
                                <Trash2 className="h-3.5 w-3.5" />
                              </Button>
                            </>
                          )}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>

        {/* Pagination */}
        {total > limit && (
          <div className="flex items-center justify-between">
            <div className="text-sm text-muted-foreground">
              Page {page} of {Math.ceil(total / limit)} · {total} total
            </div>
            <div className="flex gap-2">
              <Button
                size="sm"
                variant="outline"
                disabled={page === 1 || loading}
                onClick={() => setPage((p) => Math.max(1, p - 1))}
              >
                Previous
              </Button>
              <Button
                size="sm"
                variant="outline"
                disabled={page * limit >= total || loading}
                onClick={() => setPage((p) => p + 1)}
              >
                Next
              </Button>
            </div>
          </div>
        )}

        {/* Pause modal */}
        {pauseTarget && (
          <div
            className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4"
            onClick={() => setPauseTarget(null)}
          >
            <Card className="max-w-md w-full" onClick={(e) => e.stopPropagation()}>
              <CardContent className="p-6 space-y-4">
                <h2 className="text-lg font-semibold">
                  Pause workspace "{pauseTarget.name}"?
                </h2>
                <p className="text-sm text-muted-foreground">
                  Users in this workspace will lose access until it is resumed.
                  Enter a reason (visible only to admins).
                </p>
                <Textarea
                  placeholder="e.g. Non-payment, abuse review, security concern"
                  value={pauseReason}
                  onChange={(e) => setPauseReason(e.target.value)}
                  rows={3}
                />
                <div className="flex justify-end gap-2">
                  <Button
                    variant="outline"
                    onClick={() => setPauseTarget(null)}
                    disabled={actionId === pauseTarget.id}
                  >
                    Cancel
                  </Button>
                  <Button
                    onClick={handlePause}
                    disabled={!pauseReason.trim() || actionId === pauseTarget.id}
                    className="bg-yellow-600 hover:bg-yellow-700 text-white"
                  >
                    {actionId === pauseTarget.id ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      'Pause Workspace'
                    )}
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Delete confirmation modal */}
        {deleteTarget && (
          <div
            className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4"
            onClick={() => setDeleteTarget(null)}
          >
            <Card className="max-w-md w-full" onClick={(e) => e.stopPropagation()}>
              <CardContent className="p-6 space-y-4">
                <div className="flex items-center gap-2 text-red-400">
                  <AlertTriangle className="h-5 w-5" />
                  <h2 className="text-lg font-semibold">Delete workspace?</h2>
                </div>
                <p className="text-sm text-muted-foreground">
                  This will soft-delete{' '}
                  <span className="font-semibold text-foreground">
                    {deleteTarget.name}
                  </span>
                  . S3 data is NOT deleted yet — a separate cleanup worker handles
                  that. You can restore within the retention window.
                </p>
                <div>
                  <p className="text-xs text-muted-foreground mb-2">
                    Type the workspace name to confirm:
                  </p>
                  <Input
                    placeholder={deleteTarget.name}
                    value={deleteConfirm}
                    onChange={(e) => setDeleteConfirm(e.target.value)}
                  />
                </div>
                <div className="flex justify-end gap-2">
                  <Button
                    variant="outline"
                    onClick={() => setDeleteTarget(null)}
                    disabled={actionId === deleteTarget.id}
                  >
                    Cancel
                  </Button>
                  <Button
                    onClick={handleDelete}
                    disabled={
                      deleteConfirm !== deleteTarget.name ||
                      actionId === deleteTarget.id
                    }
                    className="bg-red-600 hover:bg-red-700 text-white"
                  >
                    {actionId === deleteTarget.id ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      'Delete Workspace'
                    )}
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
        )}
      </div>
    </MainLayout>
  )
}
