'use client'

import { useState, useEffect, useCallback } from 'react'
import {
  Shield,
  ShieldCheck,
  ShieldAlert,
  ShieldX,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Loader2,
  ChevronDown,
  ChevronUp,
  Upload,
  Package,
  Clock,
  RefreshCw,
} from 'lucide-react'
import { MainLayout } from '@/components/layout/main-layout'
import { Card, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Checkbox } from '@/components/ui/checkbox'
import { Textarea } from '@/components/ui/textarea'
import { usePageAPI } from '@/hooks/use-page-api'
import { apiClient } from '@/lib/api-client'

// ===================================================================
// Types
// ===================================================================

interface PendingPlugin {
  id: string
  slug: string
  name: string
  version: string
  description: string | null
  source_type: string | null
  security_status: string
  created_at: string | null
  risk_score: number | null
  static_findings_count: number
  llm_summary: string | null
}

interface ScanFinding {
  type?: string
  category?: string
  severity: string
  file?: string
  line?: number
  pattern?: string
  matched_text?: string
  description?: string
  evidence?: string
}

interface ScanResult {
  id: string
  plugin_slug: string
  plugin_version: string
  static_scan_status: string | null
  static_findings: ScanFinding[] | null
  blocked_patterns_found: string[] | null
  llm_scan_status: string | null
  llm_risk_score: number | null
  llm_findings: ScanFinding[] | null
  llm_summary: string | null
  llm_model_used: string | null
  llm_tokens_used: number | null
  overall_verdict: string | null
  scanned_at: string | null
  scanned_by: string | null
}

// ===================================================================
// Helpers
// ===================================================================

function verdictBadge(verdict: string | null) {
  switch (verdict) {
    case 'safe':
      return (
        <Badge className="bg-success/10 text-success border-success/20">
          <ShieldCheck className="h-3 w-3 mr-1" />
          Safe
        </Badge>
      )
    case 'review_required':
      return (
        <Badge className="bg-warning/10 text-warning border-warning/20">
          <ShieldAlert className="h-3 w-3 mr-1" />
          Review Required
        </Badge>
      )
    case 'blocked':
      return (
        <Badge className="bg-destructive/20 text-destructive border-destructive/30">
          <ShieldX className="h-3 w-3 mr-1" />
          Blocked
        </Badge>
      )
    default:
      return (
        <Badge className="bg-secondary/50 text-muted-foreground border-border/30">
          <Shield className="h-3 w-3 mr-1" />
          Pending
        </Badge>
      )
  }
}

function severityColor(severity: string) {
  switch (severity) {
    case 'critical':
      return 'text-destructive'
    case 'high':
      return 'text-orange-400'
    case 'medium':
      return 'text-warning'
    case 'low':
      return 'text-info'
    default:
      return 'text-muted-foreground'
  }
}

function riskScoreColor(score: number | null) {
  if (score === null) return 'text-muted-foreground'
  if (score < 20) return 'text-success'
  if (score < 70) return 'text-warning'
  return 'text-destructive'
}

function formatDate(dateStr: string | null) {
  if (!dateStr) return 'N/A'
  try {
    return new Date(dateStr).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    })
  } catch {
    return dateStr
  }
}

// ===================================================================
// Component
// ===================================================================

export default function AdminPluginsPendingPage() {
  usePageAPI('admin')

  // Data
  const [plugins, setPlugins] = useState<PendingPlugin[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Selection for batch actions
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())

  // Expanded scan details
  const [expandedScanId, setExpandedScanId] = useState<string | null>(null)
  const [scanResults, setScanResults] = useState<Record<string, ScanResult>>({})
  const [scanLoading, setScanLoading] = useState<string | null>(null)

  // Action states
  const [actionLoading, setActionLoading] = useState<string | null>(null)
  const [rejectingId, setRejectingId] = useState<string | null>(null)
  const [rejectReason, setRejectReason] = useState('')
  const [batchActionLoading, setBatchActionLoading] = useState(false)

  const limit = 20

  // Fetch pending plugins
  const fetchPlugins = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await apiClient.request<{
        items: PendingPlugin[]
        total: number
        page: number
        limit: number
      }>(`/api/admin/plugins/pending?page=${page}&limit=${limit}`)
      setPlugins(data.items)
      setTotal(data.total)
      // Clear selections when page data changes so batch actions only affect visible items
      setSelectedIds(new Set())
    } catch (err: any) {
      setError(err?.message || 'Failed to load pending plugins')
    } finally {
      setLoading(false)
    }
  }, [page])

  useEffect(() => {
    fetchPlugins()
  }, [fetchPlugins])

  // Fetch scan details for a plugin
  async function fetchScanDetails(pluginId: string) {
    if (scanResults[pluginId]) {
      // Already loaded, just toggle
      setExpandedScanId(expandedScanId === pluginId ? null : pluginId)
      return
    }

    setScanLoading(pluginId)
    try {
      const scan = await apiClient.request<ScanResult>(
        `/api/admin/plugins/${pluginId}/scan`
      )
      setScanResults((prev) => ({ ...prev, [pluginId]: scan }))
      setExpandedScanId(pluginId)
    } catch (err: any) {
      console.error('Failed to load scan:', err)
    } finally {
      setScanLoading(null)
    }
  }

  // Approve a single plugin
  async function handleApprove(pluginId: string) {
    setActionLoading(pluginId)
    try {
      await apiClient.request(`/api/admin/plugins/${pluginId}/approve`, {
        method: 'POST',
      })
      // Remove from list
      setPlugins((prev) => prev.filter((p) => p.id !== pluginId))
      setTotal((prev) => prev - 1)
      setSelectedIds((prev) => {
        const next = new Set(prev)
        next.delete(pluginId)
        return next
      })
    } catch (err: any) {
      setError(err?.message || 'Failed to approve plugin')
    } finally {
      setActionLoading(null)
    }
  }

  // Reject a single plugin
  async function handleReject(pluginId: string) {
    if (!rejectReason.trim()) return

    setActionLoading(pluginId)
    try {
      await apiClient.request(`/api/admin/plugins/${pluginId}/reject`, {
        method: 'POST',
        body: { reason: rejectReason },
      })
      // Remove from list
      setPlugins((prev) => prev.filter((p) => p.id !== pluginId))
      setTotal((prev) => prev - 1)
      setSelectedIds((prev) => {
        const next = new Set(prev)
        next.delete(pluginId)
        return next
      })
      setRejectingId(null)
      setRejectReason('')
    } catch (err: any) {
      setError(err?.message || 'Failed to reject plugin')
    } finally {
      setActionLoading(null)
    }
  }

  // Batch approve
  async function handleBatchApprove() {
    if (selectedIds.size === 0) return
    setBatchActionLoading(true)
    try {
      const ids = Array.from(selectedIds)
      for (const id of ids) {
        await apiClient.request(`/api/admin/plugins/${id}/approve`, {
          method: 'POST',
        })
      }
      setPlugins((prev) => prev.filter((p) => !selectedIds.has(p.id)))
      setTotal((prev) => prev - selectedIds.size)
      setSelectedIds(new Set())
    } catch (err: any) {
      setError(err?.message || 'Failed to batch approve')
      // Refresh to get accurate state
      fetchPlugins()
    } finally {
      setBatchActionLoading(false)
    }
  }

  // Batch reject
  async function handleBatchReject() {
    if (selectedIds.size === 0) return
    const reason = window.prompt('Enter rejection reason for all selected plugins:')
    if (!reason) return

    setBatchActionLoading(true)
    try {
      const ids = Array.from(selectedIds)
      for (const id of ids) {
        await apiClient.request(`/api/admin/plugins/${id}/reject`, {
          method: 'POST',
          body: { reason },
        })
      }
      setPlugins((prev) => prev.filter((p) => !selectedIds.has(p.id)))
      setTotal((prev) => prev - selectedIds.size)
      setSelectedIds(new Set())
    } catch (err: any) {
      setError(err?.message || 'Failed to batch reject')
      fetchPlugins()
    } finally {
      setBatchActionLoading(false)
    }
  }

  // Toggle selection
  function toggleSelect(pluginId: string) {
    setSelectedIds((prev) => {
      const next = new Set(prev)
      if (next.has(pluginId)) {
        next.delete(pluginId)
      } else {
        next.add(pluginId)
      }
      return next
    })
  }

  function toggleSelectAll() {
    if (selectedIds.size === plugins.length) {
      setSelectedIds(new Set())
    } else {
      setSelectedIds(new Set(plugins.map((p) => p.id)))
    }
  }

  const totalPages = Math.ceil(total / limit)

  return (
    <MainLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold">
              Plugin <span className="gradient-text">Approval Queue</span>
            </h1>
            <p className="text-muted-foreground mt-1">
              Review, approve, or reject pending plugin submissions
            </p>
          </div>
          <div className="flex items-center gap-3">
            <Button
              variant="outline"
              size="sm"
              onClick={fetchPlugins}
              disabled={loading}
            >
              <RefreshCw className={`h-4 w-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
              Refresh
            </Button>
            <Button
              size="sm"
              onClick={() => (window.location.href = '/admin/plugins/upload')}
            >
              <Upload className="h-4 w-4 mr-2" />
              Upload Plugin
            </Button>
          </div>
        </div>

        {/* Stats Bar */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <Card className="glass-card">
            <CardContent className="p-4 flex items-center gap-3">
              <div className="p-2 rounded-lg bg-warning/10">
                <Clock className="h-5 w-5 text-warning" />
              </div>
              <div>
                <p className="text-2xl font-bold text-white">{total}</p>
                <p className="text-xs text-muted-foreground">Pending Review</p>
              </div>
            </CardContent>
          </Card>
          <Card className="glass-card">
            <CardContent className="p-4 flex items-center gap-3">
              <div className="p-2 rounded-lg bg-info/10">
                <Package className="h-5 w-5 text-info" />
              </div>
              <div>
                <p className="text-2xl font-bold text-white">{selectedIds.size}</p>
                <p className="text-xs text-muted-foreground">Selected</p>
              </div>
            </CardContent>
          </Card>
          <Card className="glass-card">
            <CardContent className="p-4 flex items-center gap-3">
              <div className="p-2 rounded-lg bg-orange-500/10">
                <Shield className="h-5 w-5 text-orange-400" />
              </div>
              <div>
                <p className="text-2xl font-bold text-white">
                  {plugins.filter((p) => p.risk_score !== null && p.risk_score >= 20).length}
                </p>
                <p className="text-xs text-muted-foreground">Flagged for Review</p>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Batch Actions */}
        {selectedIds.size > 0 && (
          <Card className="glass-card border-orange-500/30">
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <span className="text-sm text-foreground/90">
                  {selectedIds.size} plugin{selectedIds.size !== 1 ? 's' : ''} selected
                </span>
                <div className="flex gap-2">
                  <Button
                    size="sm"
                    className="bg-success hover:bg-success/80 text-white"
                    onClick={handleBatchApprove}
                    disabled={batchActionLoading}
                  >
                    {batchActionLoading ? (
                      <Loader2 className="h-4 w-4 mr-1 animate-spin" />
                    ) : (
                      <CheckCircle2 className="h-4 w-4 mr-1" />
                    )}
                    Approve All
                  </Button>
                  <Button
                    size="sm"
                    variant="destructive"
                    onClick={handleBatchReject}
                    disabled={batchActionLoading}
                  >
                    {batchActionLoading ? (
                      <Loader2 className="h-4 w-4 mr-1 animate-spin" />
                    ) : (
                      <XCircle className="h-4 w-4 mr-1" />
                    )}
                    Reject All
                  </Button>
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => setSelectedIds(new Set())}
                    disabled={batchActionLoading}
                  >
                    Clear
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Error */}
        {error && (
          <div className="flex items-start gap-2 p-3 rounded-md bg-destructive/10 border border-destructive/30 text-destructive text-sm">
            <XCircle className="h-4 w-4 mt-0.5 shrink-0" />
            <span>{error}</span>
            <button className="ml-auto text-destructive/80 hover:text-destructive/70" onClick={() => setError(null)}>
              Dismiss
            </button>
          </div>
        )}

        {/* Loading */}
        {loading && (
          <div className="flex items-center justify-center py-16">
            <Loader2 className="h-8 w-8 animate-spin text-orange-400" />
          </div>
        )}

        {/* Empty State */}
        {!loading && plugins.length === 0 && (
          <Card className="glass-card">
            <CardContent className="p-12 text-center">
              <CheckCircle2 className="h-12 w-12 mx-auto text-success mb-4" />
              <h3 className="text-lg font-medium text-white mb-2">All caught up!</h3>
              <p className="text-muted-foreground">
                No plugins pending review. Upload a new plugin to get started.
              </p>
              <Button
                className="mt-4"
                onClick={() => (window.location.href = '/admin/plugins/upload')}
              >
                <Upload className="h-4 w-4 mr-2" />
                Upload Plugin
              </Button>
            </CardContent>
          </Card>
        )}

        {/* Plugin List */}
        {!loading && plugins.length > 0 && (
          <div className="space-y-4">
            {/* Select All Header */}
            <div className="flex items-center gap-3 px-1">
              <Checkbox
                checked={selectedIds.size === plugins.length && plugins.length > 0}
                onCheckedChange={toggleSelectAll}
              />
              <span className="text-sm text-muted-foreground">Select all</span>
            </div>

            {/* Plugin Cards */}
            {plugins.map((plugin) => {
              const isExpanded = expandedScanId === plugin.id
              const scan = scanResults[plugin.id]
              const isLoadingScan = scanLoading === plugin.id
              const isActionLoading = actionLoading === plugin.id
              const isRejecting = rejectingId === plugin.id

              return (
                <Card
                  key={plugin.id}
                  className={`glass-card transition-all duration-200 ${
                    selectedIds.has(plugin.id)
                      ? 'border-orange-500/50 shadow-lg shadow-orange-500/5'
                      : 'hover:border-border'
                  }`}
                >
                  <CardContent className="p-5">
                    {/* Main Row */}
                    <div className="flex items-start gap-4">
                      {/* Checkbox */}
                      <div className="pt-1">
                        <Checkbox
                          checked={selectedIds.has(plugin.id)}
                          onCheckedChange={() => toggleSelect(plugin.id)}
                        />
                      </div>

                      {/* Plugin Info */}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-3 mb-1">
                          <h3 className="text-lg font-semibold text-white truncate">
                            {plugin.name}
                          </h3>
                          <Badge variant="outline" className="border-border text-muted-foreground shrink-0">
                            v{plugin.version}
                          </Badge>
                          {verdictBadge(
                            plugin.risk_score !== null
                              ? plugin.risk_score < 20
                                ? 'safe'
                                : plugin.risk_score < 70
                                ? 'review_required'
                                : 'blocked'
                              : null
                          )}
                        </div>

                        <p className="text-sm text-muted-foreground font-mono mb-2">
                          {plugin.slug}
                        </p>

                        {plugin.description && (
                          <p className="text-sm text-muted-foreground mb-3 line-clamp-2">
                            {plugin.description}
                          </p>
                        )}

                        {/* Meta Row */}
                        <div className="flex flex-wrap items-center gap-4 text-xs text-muted-foreground">
                          <span className="flex items-center gap-1">
                            <Clock className="h-3 w-3" />
                            {formatDate(plugin.created_at)}
                          </span>
                          {plugin.source_type && (
                            <span>
                              Source: <span className="text-muted-foreground">{plugin.source_type}</span>
                            </span>
                          )}
                          <span>
                            Static findings:{' '}
                            <span
                              className={
                                plugin.static_findings_count > 0
                                  ? 'text-warning'
                                  : 'text-success'
                              }
                            >
                              {plugin.static_findings_count}
                            </span>
                          </span>
                          {plugin.risk_score !== null && (
                            <span>
                              Risk score:{' '}
                              <span className={riskScoreColor(plugin.risk_score)}>
                                {plugin.risk_score}/100
                              </span>
                            </span>
                          )}
                        </div>

                        {/* LLM Summary */}
                        {plugin.llm_summary && (
                          <div className="mt-3 p-2 rounded bg-secondary/50 border border-secondary text-sm text-foreground/90">
                            <span className="text-xs text-muted-foreground block mb-1">LLM Summary</span>
                            {plugin.llm_summary}
                          </div>
                        )}
                      </div>

                      {/* Actions */}
                      <div className="flex flex-col gap-2 shrink-0">
                        <Button
                          size="sm"
                          className="bg-success hover:bg-success/80 text-white"
                          onClick={() => handleApprove(plugin.id)}
                          disabled={isActionLoading || batchActionLoading}
                        >
                          {isActionLoading ? (
                            <Loader2 className="h-4 w-4 animate-spin" />
                          ) : (
                            <>
                              <CheckCircle2 className="h-4 w-4 mr-1" />
                              Approve
                            </>
                          )}
                        </Button>
                        <Button
                          size="sm"
                          variant="destructive"
                          onClick={() => {
                            setRejectingId(isRejecting ? null : plugin.id)
                            setRejectReason('')
                          }}
                          disabled={isActionLoading || batchActionLoading}
                        >
                          <XCircle className="h-4 w-4 mr-1" />
                          Reject
                        </Button>
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => fetchScanDetails(plugin.id)}
                          disabled={isLoadingScan}
                        >
                          {isLoadingScan ? (
                            <Loader2 className="h-4 w-4 animate-spin" />
                          ) : (
                            <>
                              {isExpanded ? (
                                <ChevronUp className="h-4 w-4 mr-1" />
                              ) : (
                                <ChevronDown className="h-4 w-4 mr-1" />
                              )}
                              Scan Details
                            </>
                          )}
                        </Button>
                      </div>
                    </div>

                    {/* Reject Reason Input */}
                    {isRejecting && (
                      <div className="mt-4 ml-10 space-y-2">
                        <Textarea
                          placeholder="Enter rejection reason..."
                          value={rejectReason}
                          onChange={(e) => setRejectReason(e.target.value)}
                          className="bg-secondary/50 border-secondary min-h-[80px]"
                        />
                        <div className="flex gap-2">
                          <Button
                            size="sm"
                            variant="destructive"
                            onClick={() => handleReject(plugin.id)}
                            disabled={!rejectReason.trim() || isActionLoading}
                          >
                            {isActionLoading ? (
                              <Loader2 className="h-4 w-4 mr-1 animate-spin" />
                            ) : (
                              'Confirm Reject'
                            )}
                          </Button>
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => {
                              setRejectingId(null)
                              setRejectReason('')
                            }}
                          >
                            Cancel
                          </Button>
                        </div>
                      </div>
                    )}

                    {/* Expanded Scan Details */}
                    {isExpanded && scan && (
                      <div className="mt-4 ml-10 space-y-4">
                        {/* Risk Score Bar */}
                        {scan.llm_risk_score !== null && (
                          <div className="p-3 rounded bg-secondary/30 border border-secondary">
                            <div className="flex items-center justify-between mb-2">
                              <span className="text-sm text-muted-foreground">Risk Score</span>
                              <span
                                className={`text-xl font-bold ${riskScoreColor(
                                  scan.llm_risk_score
                                )}`}
                              >
                                {scan.llm_risk_score}/100
                              </span>
                            </div>
                            <div className="w-full bg-secondary rounded-full h-2">
                              <div
                                className={`h-2 rounded-full transition-all ${
                                  scan.llm_risk_score < 20
                                    ? 'bg-green-500'
                                    : scan.llm_risk_score < 70
                                    ? 'bg-yellow-500'
                                    : 'bg-red-500'
                                }`}
                                style={{ width: `${Math.min(scan.llm_risk_score, 100)}%` }}
                              />
                            </div>
                          </div>
                        )}

                        {/* Overall Verdict */}
                        <div className="flex items-center gap-3">
                          <span className="text-sm text-muted-foreground">Overall Verdict:</span>
                          {verdictBadge(scan.overall_verdict)}
                        </div>

                        {/* LLM Summary */}
                        {scan.llm_summary && (
                          <div className="p-3 rounded bg-secondary/30 border border-secondary">
                            <span className="text-xs text-muted-foreground block mb-1">
                              LLM Analysis Summary
                            </span>
                            <p className="text-sm text-foreground/90">{scan.llm_summary}</p>
                            {scan.llm_model_used && (
                              <p className="text-xs text-muted-foreground mt-2">
                                Model: {scan.llm_model_used}
                                {scan.llm_tokens_used
                                  ? ` | Tokens: ${scan.llm_tokens_used}`
                                  : ''}
                              </p>
                            )}
                          </div>
                        )}

                        {/* Static Findings */}
                        {scan.static_findings && scan.static_findings.length > 0 && (
                          <div>
                            <h4 className="text-sm font-medium text-foreground/90 mb-2">
                              Static Findings ({scan.static_findings.length})
                            </h4>
                            <div className="space-y-2 max-h-48 overflow-y-auto">
                              {scan.static_findings.map((finding, idx) => (
                                <div
                                  key={idx}
                                  className="p-2 rounded bg-secondary/50 border border-secondary text-sm"
                                >
                                  <div className="flex items-center gap-2 mb-1">
                                    <AlertTriangle
                                      className={`h-3 w-3 ${severityColor(finding.severity)}`}
                                    />
                                    <span
                                      className={`font-medium ${severityColor(finding.severity)}`}
                                    >
                                      {finding.severity}
                                    </span>
                                    <span className="text-muted-foreground">|</span>
                                    <span className="text-foreground/90">
                                      {finding.type || finding.category}
                                    </span>
                                  </div>
                                  <p className="text-foreground/90">{finding.description}</p>
                                  {finding.file && (
                                    <p className="text-xs text-muted-foreground mt-1 font-mono">
                                      {finding.file}
                                      {finding.line ? `:${finding.line}` : ''}
                                    </p>
                                  )}
                                  {finding.matched_text && (
                                    <p className="text-xs text-muted-foreground mt-1 font-mono truncate">
                                      Match: {finding.matched_text}
                                    </p>
                                  )}
                                </div>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* LLM Findings */}
                        {scan.llm_findings && scan.llm_findings.length > 0 && (
                          <div>
                            <h4 className="text-sm font-medium text-foreground/90 mb-2">
                              LLM Findings ({scan.llm_findings.length})
                            </h4>
                            <div className="space-y-2 max-h-48 overflow-y-auto">
                              {scan.llm_findings.map((finding, idx) => (
                                <div
                                  key={idx}
                                  className="p-2 rounded bg-secondary/50 border border-secondary text-sm"
                                >
                                  <div className="flex items-center gap-2 mb-1">
                                    <AlertTriangle
                                      className={`h-3 w-3 ${severityColor(finding.severity)}`}
                                    />
                                    <span
                                      className={`font-medium ${severityColor(finding.severity)}`}
                                    >
                                      {finding.severity}
                                    </span>
                                    <span className="text-muted-foreground">|</span>
                                    <span className="text-foreground/90">
                                      {finding.category || finding.type}
                                    </span>
                                  </div>
                                  <p className="text-foreground/90">{finding.description}</p>
                                  {finding.file && (
                                    <p className="text-xs text-muted-foreground mt-1 font-mono">
                                      {finding.file}
                                    </p>
                                  )}
                                  {finding.evidence && (
                                    <p className="text-xs text-muted-foreground mt-1 font-mono truncate">
                                      Evidence: {finding.evidence}
                                    </p>
                                  )}
                                </div>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* No findings */}
                        {(!scan.static_findings || scan.static_findings.length === 0) &&
                          (!scan.llm_findings || scan.llm_findings.length === 0) && (
                            <div className="text-sm text-muted-foreground flex items-center gap-2">
                              <ShieldCheck className="h-4 w-4 text-success" />
                              No security findings detected
                            </div>
                          )}
                      </div>
                    )}
                  </CardContent>
                </Card>
              )
            })}

            {/* Pagination */}
            {totalPages > 1 && (
              <div className="flex items-center justify-center gap-2 pt-4">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setPage((p) => Math.max(1, p - 1))}
                  disabled={page <= 1}
                >
                  Previous
                </Button>
                <span className="text-sm text-muted-foreground px-3">
                  Page {page} of {totalPages}
                </span>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                  disabled={page >= totalPages}
                >
                  Next
                </Button>
              </div>
            )}
          </div>
        )}
      </div>
    </MainLayout>
  )
}
