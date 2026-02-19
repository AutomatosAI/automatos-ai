'use client'

import { useState, useEffect, useCallback } from 'react'
import { apiClient } from '@/lib/api-client'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Textarea } from '@/components/ui/textarea'
import {
  FileText,
  Search,
  ChevronRight,
  History,
  Play,
  RotateCcw,
  Trash2,
  Check,
  Shield,
  Zap,
  BarChart3,
  Loader2,
  ArrowLeft,
  Save,
  X,
} from 'lucide-react'
import { cn } from '@/lib/utils'

// =====================================================================
// Types
// =====================================================================

interface PromptVersion {
  id: string
  version_number: number
  content: string
  change_note: string | null
  status: string
  created_by: string | null
  created_at: string
  eval_scores: Record<string, any> | null
}

interface SystemPrompt {
  id: string
  slug: string
  display_name: string
  category: string
  description: string | null
  variables: Record<string, string> | null
  is_active: boolean
  futureagi_eval_enabled: boolean
  active_version_number: number | null
  active_content: string | null
  active_eval_scores: Record<string, any> | null
  version_count: number
  created_at: string
  updated_at: string
}

interface AssessmentRun {
  id: string
  prompt_id: string
  version_id: string | null
  run_type: string
  status: string
  scores: Record<string, any> | null
  error_message: string | null
  started_at: string | null
  completed_at: string | null
  created_at: string
}

interface Category {
  category: string
  count: number
}

// =====================================================================
// Styling
// =====================================================================

const CATEGORY_COLORS: Record<string, string> = {
  personality: 'bg-violet-500/15 text-violet-400 border-violet-500/30',
  orchestrator: 'bg-blue-500/15 text-blue-400 border-blue-500/30',
  specialized: 'bg-amber-500/15 text-amber-400 border-amber-500/30',
  persona: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30',
}

const STATUS_COLORS: Record<string, string> = {
  active: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30',
  draft: 'bg-yellow-500/15 text-yellow-400 border-yellow-500/30',
  archived: 'bg-zinc-500/15 text-zinc-400 border-zinc-500/30',
}

// =====================================================================
// Component
// =====================================================================

export function SystemPromptsTab() {
  const [prompts, setPrompts] = useState<SystemPrompt[]>([])
  const [categories, setCategories] = useState<Category[]>([])
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedPrompt, setSelectedPrompt] = useState<SystemPrompt | null>(null)
  const [versions, setVersions] = useState<PromptVersion[]>([])
  const [assessmentRuns, setAssessmentRuns] = useState<AssessmentRun[]>([])
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [editContent, setEditContent] = useState('')
  const [editNote, setEditNote] = useState('')
  const [showEditor, setShowEditor] = useState(false)
  const [activeTab, setActiveTab] = useState<'content' | 'versions' | 'assessments'>('content')
  const [assessLoading, setAssessLoading] = useState<string | null>(null)

  // ---------------------------------------------------------------
  // Data fetching
  // ---------------------------------------------------------------

  const fetchPrompts = useCallback(async () => {
    try {
      setLoading(true)
      const params = new URLSearchParams()
      if (selectedCategory) params.set('category', selectedCategory)
      if (searchQuery) params.set('search', searchQuery)
      const data = await apiClient.request(`/api/admin/prompts?${params}`)
      setPrompts(data)
    } catch (err) {
      console.error('Failed to fetch prompts:', err)
    } finally {
      setLoading(false)
    }
  }, [selectedCategory, searchQuery])

  const fetchCategories = useCallback(async () => {
    try {
      const data = await apiClient.request('/api/admin/prompts/categories')
      setCategories(data)
    } catch (err) {
      console.error('Failed to fetch categories:', err)
    }
  }, [])

  const fetchVersions = useCallback(async (promptId: string) => {
    try {
      const data = await apiClient.request(`/api/admin/prompts/${promptId}/versions`)
      setVersions(data)
    } catch (err) {
      console.error('Failed to fetch versions:', err)
    }
  }, [])

  const fetchAssessmentRuns = useCallback(async (promptId: string) => {
    try {
      const data = await apiClient.request(`/api/admin/prompts/${promptId}/assessment-runs`)
      setAssessmentRuns(data)
    } catch (err) {
      console.error('Failed to fetch assessment runs:', err)
    }
  }, [])

  useEffect(() => { fetchCategories() }, [fetchCategories])
  useEffect(() => { fetchPrompts() }, [fetchPrompts])

  // Auto-poll assessment runs when any are pending/running
  useEffect(() => {
    const hasPending = assessmentRuns.some(r => r.status === 'pending' || r.status === 'running')
    if (!hasPending || !selectedPrompt) return
    const interval = setInterval(() => {
      fetchAssessmentRuns(selectedPrompt.id)
    }, 3000)
    return () => clearInterval(interval)
  }, [assessmentRuns, selectedPrompt, fetchAssessmentRuns])

  // ---------------------------------------------------------------
  // Select prompt
  // ---------------------------------------------------------------

  const selectPrompt = (prompt: SystemPrompt) => {
    setSelectedPrompt(prompt)
    setEditContent(prompt.active_content || '')
    setEditNote('')
    setShowEditor(false)
    setActiveTab('content')
    fetchVersions(prompt.id)
    fetchAssessmentRuns(prompt.id)
  }

  // ---------------------------------------------------------------
  // Actions
  // ---------------------------------------------------------------

  const refreshSelected = async () => {
    if (!selectedPrompt) return
    await fetchPrompts()
    await fetchVersions(selectedPrompt.id)
    const updated = await apiClient.request(`/api/admin/prompts/${selectedPrompt.id}`)
    setSelectedPrompt(updated)
    setEditContent(updated.active_content || '')
  }

  const saveVersion = async (activate: boolean) => {
    if (!selectedPrompt || !editContent.trim()) return
    setSaving(true)
    try {
      await apiClient.request(`/api/admin/prompts/${selectedPrompt.id}/versions?activate=${activate}`, {
        method: 'POST',
        body: JSON.stringify({ content: editContent, change_note: editNote || null }),
      })
      setShowEditor(false)
      await refreshSelected()
    } catch (err) {
      console.error('Failed to save version:', err)
    } finally {
      setSaving(false)
    }
  }

  const activateVersion = async (versionId: string) => {
    if (!selectedPrompt) return
    try {
      await apiClient.request(`/api/admin/prompts/${selectedPrompt.id}/versions/${versionId}/activate`, { method: 'POST' })
      await refreshSelected()
    } catch (err) {
      console.error('Failed to activate version:', err)
    }
  }

  const rollback = async () => {
    if (!selectedPrompt) return
    try {
      await apiClient.request(`/api/admin/prompts/${selectedPrompt.id}/rollback`, { method: 'POST' })
      await refreshSelected()
    } catch (err) {
      console.error('Failed to rollback:', err)
    }
  }

  const deleteVersion = async (versionId: string) => {
    if (!selectedPrompt) return
    try {
      await apiClient.request(`/api/admin/prompts/${selectedPrompt.id}/versions/${versionId}`, { method: 'DELETE' })
      await fetchVersions(selectedPrompt.id)
    } catch (err) {
      console.error('Failed to delete version:', err)
    }
  }

  const toggleFutureAGI = async () => {
    if (!selectedPrompt) return
    try {
      const updated = await apiClient.request(`/api/admin/prompts/${selectedPrompt.id}/futureagi-toggle`, {
        method: 'PATCH',
      })
      setSelectedPrompt(updated as SystemPrompt)
    } catch (err) {
      console.error('Failed to toggle FutureAGI:', err)
    }
  }

  const applyOptimizedPrompt = async (optimizedText: string) => {
    if (!selectedPrompt || !optimizedText) return
    setSaving(true)
    try {
      await apiClient.request(`/api/admin/prompts/${selectedPrompt.id}/versions?activate=false`, {
        method: 'POST',
        body: JSON.stringify({
          content: optimizedText,
          change_note: 'Auto-generated by FutureAGI optimizer',
        }),
      })
      await refreshSelected()
      setActiveTab('versions')
    } catch (err) {
      console.error('Failed to apply optimized prompt:', err)
    } finally {
      setSaving(false)
    }
  }

  const triggerAssessment = async (runType: string) => {
    if (!selectedPrompt) return
    setAssessLoading(runType)
    try {
      await apiClient.request(`/api/admin/prompts/${selectedPrompt.id}/assess`, {
        method: 'POST',
        body: JSON.stringify({ run_type: runType }),
      })
      setActiveTab('assessments')
      await fetchAssessmentRuns(selectedPrompt.id)
    } catch (err) {
      console.error('Failed to trigger assessment:', err)
    } finally {
      setAssessLoading(null)
    }
  }

  // ---------------------------------------------------------------
  // Render: List view
  // ---------------------------------------------------------------

  if (!selectedPrompt) {
    return (
      <div className="space-y-4">
        {/* Filters */}
        <div className="flex items-center gap-3 flex-wrap">
          <div className="relative flex-1 max-w-sm">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <Input
              placeholder="Search prompts..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-9"
            />
          </div>
          <div className="flex items-center gap-1.5 flex-wrap">
            <button
              onClick={() => setSelectedCategory(null)}
              className={cn(
                'px-3 py-1.5 rounded-lg text-xs font-medium transition-colors',
                !selectedCategory ? 'bg-primary/15 text-primary' : 'text-muted-foreground hover:text-foreground'
              )}
            >
              All ({prompts.length})
            </button>
            {categories.map((cat) => (
              <button
                key={cat.category}
                onClick={() => setSelectedCategory(cat.category)}
                className={cn(
                  'px-3 py-1.5 rounded-lg text-xs font-medium transition-colors',
                  selectedCategory === cat.category ? 'bg-primary/15 text-primary' : 'text-muted-foreground hover:text-foreground'
                )}
              >
                {cat.category} ({cat.count})
              </button>
            ))}
          </div>
        </div>

        {/* List */}
        {loading ? (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
          </div>
        ) : prompts.length === 0 ? (
          <div className="text-center py-12 text-muted-foreground">No prompts found</div>
        ) : (
          <div className="grid gap-3">
            {prompts.map((prompt) => (
              <button
                key={prompt.id}
                onClick={() => selectPrompt(prompt)}
                className="flex items-center justify-between w-full p-4 rounded-xl border border-border/50 hover:border-primary/30 hover:bg-secondary/20 transition-all text-left group"
              >
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="font-medium text-sm">{prompt.display_name}</span>
                    <Badge variant="outline" className={cn('text-[10px]', CATEGORY_COLORS[prompt.category] || '')}>
                      {prompt.category}
                    </Badge>
                    {prompt.active_version_number && (
                      <span className="text-xs text-muted-foreground">v{prompt.active_version_number}</span>
                    )}
                  </div>
                  <p className="text-xs text-muted-foreground truncate">{prompt.description || prompt.slug}</p>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-xs text-muted-foreground">
                    {prompt.version_count} version{prompt.version_count !== 1 ? 's' : ''}
                  </span>
                  <ChevronRight className="w-4 h-4 text-muted-foreground group-hover:text-primary transition-colors" />
                </div>
              </button>
            ))}
          </div>
        )}
      </div>
    )
  }

  // ---------------------------------------------------------------
  // Render: Detail view
  // ---------------------------------------------------------------

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center gap-3">
        <Button variant="ghost" size="icon" onClick={() => setSelectedPrompt(null)}>
          <ArrowLeft className="w-4 h-4" />
        </Button>
        <FileText className="w-5 h-5 text-amber-400" />
        <h2 className="text-base font-semibold">{selectedPrompt.display_name}</h2>
        <Badge variant="outline" className={CATEGORY_COLORS[selectedPrompt.category] || ''}>
          {selectedPrompt.category}
        </Badge>
      </div>

      {/* Sub-tabs */}
      <div className="flex items-center gap-1">
        {(['content', 'versions', 'assessments'] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={cn(
              'px-4 py-2 rounded-lg text-sm font-medium transition-colors',
              activeTab === tab ? 'bg-primary/15 text-primary' : 'text-muted-foreground hover:text-foreground'
            )}
          >
            {tab === 'content' && <FileText className="w-3.5 h-3.5 inline mr-1.5" />}
            {tab === 'versions' && <History className="w-3.5 h-3.5 inline mr-1.5" />}
            {tab === 'assessments' && <BarChart3 className="w-3.5 h-3.5 inline mr-1.5" />}
            {tab.charAt(0).toUpperCase() + tab.slice(1)}
          </button>
        ))}
      </div>

      {/* CONTENT */}
      {activeTab === 'content' && (
        <div className="space-y-4">
          <div className="flex items-center gap-3 flex-wrap">
            <code className="text-xs bg-secondary/50 px-2 py-1 rounded">{selectedPrompt.slug}</code>
            {selectedPrompt.variables && (
              <div className="flex gap-1.5 flex-wrap">
                {Object.keys(selectedPrompt.variables).map((v) => (
                  <Badge key={v} variant="outline" className="text-[10px]">
                    {'{' + v + '}'}
                  </Badge>
                ))}
              </div>
            )}
          </div>

          {showEditor ? (
            <div className="space-y-3">
              <Textarea
                value={editContent}
                onChange={(e) => setEditContent(e.target.value)}
                rows={16}
                className="font-mono text-sm"
                placeholder="Enter prompt content..."
              />
              <Input
                value={editNote}
                onChange={(e) => setEditNote(e.target.value)}
                placeholder="Change note (optional)"
                className="max-w-md"
              />
              <div className="flex items-center gap-2">
                <Button onClick={() => saveVersion(false)} disabled={saving || !editContent.trim()} variant="outline" size="sm">
                  {saving ? <Loader2 className="w-3.5 h-3.5 animate-spin mr-1.5" /> : <Save className="w-3.5 h-3.5 mr-1.5" />}
                  Save as Draft
                </Button>
                <Button onClick={() => saveVersion(true)} disabled={saving || !editContent.trim()} size="sm">
                  {saving ? <Loader2 className="w-3.5 h-3.5 animate-spin mr-1.5" /> : <Play className="w-3.5 h-3.5 mr-1.5" />}
                  Save & Activate
                </Button>
                <Button variant="ghost" size="sm" onClick={() => { setShowEditor(false); setEditContent(selectedPrompt.active_content || '') }}>
                  <X className="w-3.5 h-3.5 mr-1.5" />Cancel
                </Button>
              </div>
            </div>
          ) : (
            <div className="space-y-3">
              <pre className="p-4 rounded-xl bg-secondary/30 border border-border/50 text-sm whitespace-pre-wrap font-mono leading-relaxed max-h-[500px] overflow-y-auto">
                {selectedPrompt.active_content || '(no active content)'}
              </pre>
              <div className="flex items-center gap-2">
                <Button size="sm" onClick={() => setShowEditor(true)}>
                  <FileText className="w-3.5 h-3.5 mr-1.5" />Edit Prompt
                </Button>
                <Button size="sm" variant="outline" onClick={rollback}>
                  <RotateCcw className="w-3.5 h-3.5 mr-1.5" />Rollback
                </Button>
              </div>
            </div>
          )}

          {selectedPrompt.active_eval_scores && (
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">Quality Scores</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                  {Object.entries(selectedPrompt.active_eval_scores).map(([key, val]) => {
                    const score = typeof val === 'object' ? val?.score : val
                    const pct = score != null ? Math.round(score * 100) : null
                    return (
                      <div key={key} className="space-y-1">
                        <div className="flex justify-between text-xs">
                          <span className="text-muted-foreground">{key.replace(/_/g, ' ')}</span>
                          <span className="font-medium">{pct != null ? `${pct}%` : 'N/A'}</span>
                        </div>
                        <div className="h-1.5 bg-secondary rounded-full overflow-hidden">
                          <div
                            className={cn(
                              'h-full rounded-full transition-all',
                              pct != null && pct >= 70 ? 'bg-emerald-500' : pct != null && pct >= 40 ? 'bg-yellow-500' : 'bg-red-500'
                            )}
                            style={{ width: `${pct ?? 0}%` }}
                          />
                        </div>
                      </div>
                    )
                  })}
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      )}

      {/* VERSIONS */}
      {activeTab === 'versions' && (
        <div className="space-y-2">
          {versions.length === 0 ? (
            <p className="text-sm text-muted-foreground py-8 text-center">No versions</p>
          ) : (
            versions.map((v) => (
              <div key={v.id} className="flex items-center justify-between p-3 rounded-xl border border-border/50 hover:border-border transition-colors">
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium">v{v.version_number}</span>
                    <Badge variant="outline" className={cn('text-[10px]', STATUS_COLORS[v.status] || '')}>{v.status}</Badge>
                    {v.change_note && <span className="text-xs text-muted-foreground truncate">{v.change_note}</span>}
                  </div>
                  <p className="text-xs text-muted-foreground mt-0.5">
                    {new Date(v.created_at).toLocaleString()}{v.created_by && ` by ${v.created_by}`}
                  </p>
                </div>
                <div className="flex items-center gap-1.5">
                  {v.status !== 'active' && (
                    <Button variant="ghost" size="icon" className="h-7 w-7" onClick={() => activateVersion(v.id)} title="Activate">
                      <Check className="w-3.5 h-3.5" />
                    </Button>
                  )}
                  {v.status === 'draft' && (
                    <Button variant="ghost" size="icon" className="h-7 w-7 text-destructive" onClick={() => deleteVersion(v.id)} title="Delete draft">
                      <Trash2 className="w-3.5 h-3.5" />
                    </Button>
                  )}
                </div>
              </div>
            ))
          )}
        </div>
      )}

      {/* ASSESSMENTS */}
      {activeTab === 'assessments' && (
        <div className="space-y-4">
          {/* Live eval toggle */}
          <div className="flex items-center justify-between p-3 rounded-xl border border-border/50 bg-secondary/10">
            <div>
              <div className="text-sm font-medium">FutureAGI Live Scoring</div>
              <div className="text-xs text-muted-foreground">Score every real chat message using this prompt</div>
            </div>
            <button
              onClick={toggleFutureAGI}
              className={cn(
                'relative inline-flex h-6 w-11 items-center rounded-full transition-colors',
                selectedPrompt?.futureagi_eval_enabled ? 'bg-emerald-500' : 'bg-zinc-600'
              )}
            >
              <span className={cn(
                'inline-block h-4 w-4 rounded-full bg-white transition-transform',
                selectedPrompt?.futureagi_eval_enabled ? 'translate-x-6' : 'translate-x-1'
              )} />
            </button>
          </div>

          <div className="flex items-center gap-2">
            <Button size="sm" onClick={() => triggerAssessment('assess')} disabled={!!assessLoading}>
              {assessLoading === 'assess' ? <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" /> : <BarChart3 className="w-3.5 h-3.5 mr-1.5" />}
              Score Quality
            </Button>
            <Button size="sm" variant="outline" onClick={() => triggerAssessment('optimize')} disabled={!!assessLoading}>
              {assessLoading === 'optimize' ? <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" /> : <Zap className="w-3.5 h-3.5 mr-1.5" />}
              Optimize
            </Button>
            <Button size="sm" variant="outline" onClick={() => triggerAssessment('safety')} disabled={!!assessLoading}>
              {assessLoading === 'safety' ? <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" /> : <Shield className="w-3.5 h-3.5 mr-1.5" />}
              Safety Scan
            </Button>
          </div>

          {assessmentRuns.length === 0 ? (
            <p className="text-sm text-muted-foreground py-8 text-center">No assessment runs yet. Click a button above to start.</p>
          ) : (
            <div className="space-y-2">
              {assessmentRuns.map((run) => (
                <div key={run.id} className="p-3 rounded-xl border border-border/50">
                  <div className="flex items-center justify-between mb-1">
                    <div className="flex items-center gap-2">
                      <Badge variant="outline" className="text-[10px]">{run.run_type}</Badge>
                      <Badge
                        variant="outline"
                        className={cn(
                          'text-[10px]',
                          run.status === 'completed' ? 'bg-emerald-500/15 text-emerald-400' :
                          run.status === 'running' ? 'bg-blue-500/15 text-blue-400' :
                          run.status === 'failed' ? 'bg-red-500/15 text-red-400' :
                          'bg-zinc-500/15 text-zinc-400'
                        )}
                      >
                        {run.status === 'running' && <Loader2 className="w-3 h-3 animate-spin mr-1" />}
                        {run.status}
                      </Badge>
                    </div>
                    <span className="text-xs text-muted-foreground">{new Date(run.created_at).toLocaleString()}</span>
                  </div>
                  {run.error_message && <p className="text-xs text-destructive mt-1">{run.error_message}</p>}

                  {/* Assess / Live results */}
                  {run.scores && (run.run_type === 'assess' || run.run_type === 'live') && run.scores.scores && (
                    <div className="mt-2 space-y-2">
                      {Object.entries(run.scores.scores as Record<string, any>).map(([key, val]) => {
                        const v = val as any
                        const passed = v?.passed
                        const score = v?.score
                        const pct = score != null ? Math.round(Number(score) * 100) : null
                        return (
                          <div key={key} className="text-xs">
                            <div className="flex items-center gap-1.5 mb-0.5">
                              <span className={cn('inline-block w-1.5 h-1.5 rounded-full', passed ? 'bg-emerald-400' : passed === false ? 'bg-amber-400' : 'bg-zinc-400')} />
                              <span className="font-medium">{key.replace(/_/g, ' ')}</span>
                              {pct != null && <span className="text-muted-foreground">({pct}%)</span>}
                            </div>
                            {v?.reason && <p className="text-muted-foreground ml-3 line-clamp-2">{v.reason}</p>}
                          </div>
                        )
                      })}
                    </div>
                  )}

                  {/* Safety results */}
                  {run.scores && run.run_type === 'safety' && (
                    <div className="mt-2 space-y-1.5">
                      {run.scores.safe != null && (
                        <div className="flex items-center gap-1.5 text-xs font-medium">
                          <span className={cn('inline-block w-2 h-2 rounded-full', run.scores.safe ? 'bg-emerald-400' : 'bg-red-400')} />
                          {run.scores.safe ? 'All checks passed' : 'Issues detected'}
                        </div>
                      )}
                      {run.scores.checks && Object.entries(run.scores.checks as Record<string, any>).map(([key, val]) => {
                        const v = val as any
                        return (
                          <div key={key} className="text-xs">
                            <div className="flex items-center gap-1.5">
                              <span className={cn('inline-block w-1.5 h-1.5 rounded-full', v?.safe ? 'bg-emerald-400' : v?.safe === false ? 'bg-red-400' : 'bg-zinc-400')} />
                              <span className="font-medium">{key.replace(/_/g, ' ')}</span>
                              {v?.safe != null && <span className="text-muted-foreground">({v.safe ? 'safe' : 'flagged'})</span>}
                            </div>
                            {v?.reason && <p className="text-muted-foreground ml-3 line-clamp-2">{v.reason}</p>}
                            {v?.error && <p className="text-amber-400 ml-3">{v.error}</p>}
                          </div>
                        )
                      })}
                    </div>
                  )}

                  {/* Optimize results */}
                  {run.scores && run.run_type === 'optimize' && (
                    <div className="mt-2 text-xs space-y-2">
                      {run.scores.error && (
                        <p className="text-red-400">{run.scores.error}</p>
                      )}
                      {run.scores.status === 'completed' && run.scores.optimized_prompt && (
                        <div>
                          <div className="flex items-center justify-between mb-1">
                            <div className="flex items-center gap-2">
                              {run.scores.best_score != null && run.scores.best_score >= 0 ? (
                                <>
                                  <span className="font-medium text-emerald-400">Optimized prompt ready</span>
                                  <span className="text-muted-foreground">
                                    Score: {(run.scores.best_score * 100).toFixed(0)}%
                                    {run.scores.initial_score != null && run.scores.initial_score >= 0 && (
                                      <> (was {(run.scores.initial_score * 100).toFixed(0)}%)</>
                                    )}
                                  </span>
                                </>
                              ) : (
                                <span className="font-medium text-amber-400">Optimization completed (scoring failed — template variables may have interfered)</span>
                              )}
                            </div>
                            <Button
                              size="sm"
                              variant="outline"
                              className="text-xs h-7"
                              disabled={saving}
                              onClick={() => applyOptimizedPrompt(run.scores!.optimized_prompt)}
                            >
                              {saving ? <Loader2 className="w-3 h-3 mr-1 animate-spin" /> : <Check className="w-3 h-3 mr-1" />}
                              Apply as Draft
                            </Button>
                          </div>
                          <p className="p-2.5 rounded-lg bg-secondary/30 whitespace-pre-wrap max-h-48 overflow-y-auto border border-border/30">
                            {run.scores.optimized_prompt}
                          </p>
                          {run.scores.duration && (
                            <p className="text-muted-foreground mt-1">
                              {run.scores.algorithm} / {run.scores.rounds} rounds / {run.scores.duration}s
                            </p>
                          )}
                        </div>
                      )}
                      {run.scores.status === 'submitted' && (
                        <div>
                          <p className="text-blue-400">{run.scores.message || 'Optimization submitted to FutureAGI'}</p>
                          {run.scores.improve_id && (
                            <p className="text-muted-foreground">Job ID: {run.scores.improve_id}</p>
                          )}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
