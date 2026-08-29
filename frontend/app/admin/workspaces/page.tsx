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
import { SaasOnlyNotice } from '@/components/local/saas-only-notice'
import { isRouteAvailableInEdition } from '@/lib/auth-edition'

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
      <Badge className="bg-destructive/20 text-destructive border-destructive/30">
        Deleted
      </Badge>
    )
  }
  if (w.paused_at) {
    return (
      <Badge className="bg-warning/10 text-warning border-warning/20">
        Disabled
      </Badge>
    )
  }
  return (
    <Badge className="bg-success/10 text-success border-success/20">
      Enabled
    </Badge>
  )
}

// ===================================================================
// Page
// ===================================================================

function AdminWorkspacesConsole() {
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
              Manage all workspaces — enable, disable, delete, inspect usage.
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
              <div className="text-xs text-muted-foreground">Disabled / Deleted</div>
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
          <Card className="border-destructive/30 bg-destructive/5">
            <CardContent className="p-4 flex items-center gap-2 text-destructive">
              <AlertTriangle className="h-4 w-4" />
              {error}
            </CardContent>
          </Card>
        )}

        {/* Table */}
        <Card>
          <CardContent className="p-0">
            <div className="overflow-x-auto">
              <table className="w-full text-sm table-fixed">
                <thead className="bg-muted/40 text-xs uppercase text-muted-foreground">
                  <tr>
                    <th className="text-left px-3 py-2.5 w-[260px]">Workspace</th>
                    <th className="text-left px-3 py-2.5 w-[200px]">Owner</th>
                    <th className="text-left px-3 py-2.5 w-[110px]">Plan</th>
                    <th className="text-left px-3 py-2.5 w-[90px]">State</th>
                    <th className="text-right px-2 py-2.5 w-[52px]">
                      <Users className="h-3.5 w-3.5 inline" />
                    </th>
                    <th className="text-right px-2 py-2.5 w-[52px]">
                      <FileText className="h-3.5 w-3.5 inline" />
                    </th>
                    <th className="text-right px-2 py-2.5 w-[68px]">
                      <HardDrive className="h-3.5 w-3.5 inline" />
                    </th>
                    <th className="text-right px-2 py-2.5 w-[52px]">
                      <MessageSquare className="h-3.5 w-3.5 inline" />
                    </th>
                    <th className="text-left px-3 py-2.5 w-[90px]">Created</th>
                    <th className="text-right px-3 py-2.5 w-[100px]">Actions</th>
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
                      <td className="px-3 py-2.5">
                        <div className="font-medium truncate" title={w.name}>{w.name}</div>
                        <div className="flex items-center gap-1 mt-0.5">
                          <code className="text-[10px] font-mono text-muted-foreground truncate">
                            {w.id}
                          </code>
                          <button
                            onClick={() => copyWorkspaceId(w.id)}
                            className="opacity-60 hover:opacity-100 shrink-0"
                            title="Copy UUID"
                          >
                            {copiedId === w.id ? (
                              <Check className="h-3 w-3 text-success" />
                            ) : (
                              <Copy className="h-3 w-3" />
                            )}
                          </button>
                        </div>
                      </td>
                      <td className="px-3 py-2.5">
                        <div className="truncate" title={w.owner_name || ''}>
                          {w.owner_name || '—'}
                        </div>
                        <div
                          className="text-xs text-muted-foreground truncate"
                          title={w.owner_email || ''}
                        >
                          {w.owner_email || '—'}
                        </div>
                      </td>
                      <td className="px-3 py-2.5">
                        <Badge variant="outline" className="capitalize">
                          {w.plan || '—'}
                        </Badge>
                        {w.is_personal && (
                          <Badge variant="outline" className="ml-1 text-[10px]">
                            personal
                          </Badge>
                        )}
                      </td>
                      <td className="px-3 py-2.5">{stateBadge(w)}</td>
                      <td className="px-2 py-2.5 text-right tabular-nums">
                        {w.agents_count}
                      </td>
                      <td className="px-2 py-2.5 text-right tabular-nums">
                        {w.documents_count}
                      </td>
                      <td className="px-2 py-2.5 text-right tabular-nums whitespace-nowrap">
                        {formatBytes(w.storage_bytes)}
                      </td>
                      <td className="px-2 py-2.5 text-right tabular-nums">
                        {w.chats_count}
                      </td>
                      <td className="px-3 py-2.5 text-xs text-muted-foreground whitespace-nowrap">
                        {formatDate(w.created_at)}
                      </td>
                      <td className="px-3 py-2.5 text-right">
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
                                  title="Enable"
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
                                  title="Disable"
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
                                className="text-destructive hover:text-destructive/80"
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
                  Disable workspace "{pauseTarget.name}"?
                </h2>
                <p className="text-sm text-muted-foreground">
                  The owner will lose login and API access, and any scheduled
                  playbooks will refuse to run until the workspace is enabled
                  again. Enter a reason (visible only to admins).
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
                    className="bg-warning hover:bg-warning/80 text-white"
                  >
                    {actionId === pauseTarget.id ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      'Disable Workspace'
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
                <div className="flex items-center gap-2 text-destructive">
                  <AlertTriangle className="h-5 w-5" />
                  <h2 className="text-lg font-semibold">Delete workspace?</h2>
                </div>
                <p className="text-sm text-muted-foreground">
                  This permanently deletes{' '}
                  <span className="font-semibold text-foreground">
                    {deleteTarget.name}
                  </span>
                  , wipes all S3 files under this workspace, removes the
                  owner's Clerk account, and cascades every workspace-scoped
                  database row. <span className="text-destructive font-semibold">
                  This cannot be undone.</span>
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
                    className="bg-destructive hover:bg-destructive/80 text-white"
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

// PRD-233 S7: Workspace Admin is a hosted-edition surface (cross-tenant
// administration). The console above is untouched; in local the route renders
// the shared notice instead — before any of the console's hooks or admin
// fetches run. Hidden by the seam's explicit list, not by role (the local
// operator is super_admin by design).
export default function AdminWorkspacesPage() {
  if (!isRouteAvailableInEdition('/admin/workspaces')) {
    return (
      <MainLayout>
        <SaasOnlyNotice surface="Workspace Admin" />
      </MainLayout>
    )
  }
  return <AdminWorkspacesConsole />
}
