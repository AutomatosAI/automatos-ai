/**
 * Skill Editor Modal — view / edit / new modes in one component.
 *
 * - 'view'  : read-only render of an existing skill
 * - 'edit'  : edit a workspace-owned skill, OR fork-on-save for marketplace skills
 * - 'new'   : create a workspace skill from pasted content or a dropped .md file
 *
 * Save runs through the workspace skills API which calls plugin_security_scanner.
 * Critical findings hard-block; high findings show inline warnings + an
 * "I understand, save anyway" toggle.
 */

'use client'

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  AlertTriangle,
  Brain,
  Eye,
  GitBranch,
  Loader2,
  Pencil,
  Plus,
  Save,
  ShieldAlert,
  Trash2,
  Upload,
  UserPlus,
} from 'lucide-react'

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { Textarea } from '@/components/ui/textarea'
import { Checkbox } from '@/components/ui/checkbox'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { toast } from 'sonner'
import { useWorkspace } from '@/components/workspace-provider'
import { useAgents } from '@/hooks/use-agent-api'
import {
  useSkillsApi,
  SkillScanError,
  type ScannerFinding,
  type WorkspaceSkillContent,
} from '@/hooks/use-skills-api'

export type SkillEditorMode = 'view' | 'edit' | 'new'

interface Props {
  open: boolean
  mode: SkillEditorMode
  skillId: number | null
  onClose: (didChange: boolean) => void
}

const MAX_UPLOAD_BYTES = 256 * 1024 // 256KB — plenty for SKILL.md

const SAMPLE_SKELETON = `---
name: my-new-skill
description: One-line summary of what this skill does
category: marketing
tags:
  - example
version: 1.0.0
---

# My New Skill

## When to use
Describe the situations where the agent should reach for this skill.

## How it works
Step-by-step instructions the agent should follow.

## Example
A concrete worked example so the agent can pattern-match.
`

export function SkillEditorModal({ open, mode: initialMode, skillId, onClose }: Props) {
  const { workspace } = useWorkspace()
  const { data: agents = [] } = useAgents()
  const {
    getWorkspaceSkillContent,
    createWorkspaceSkill,
    updateWorkspaceSkill,
    deleteWorkspaceSkill,
    assignSkillsToAgent,
  } = useSkillsApi()

  const [mode, setMode] = useState<SkillEditorMode>(initialMode)
  const [skill, setSkill] = useState<WorkspaceSkillContent | null>(null)
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [category, setCategory] = useState('')
  const [tagsCsv, setTagsCsv] = useState('')
  const [content, setContent] = useState('')
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [findings, setFindings] = useState<ScannerFinding[]>([])
  const [acknowledgeWarnings, setAcknowledgeWarnings] = useState(false)
  const [confirmDelete, setConfirmDelete] = useState(false)
  const [assignAgentId, setAssignAgentId] = useState<string>('')
  const fileInputRef = useRef<HTMLInputElement>(null)

  const isView = mode === 'view'
  const isNew = mode === 'new'
  const willFork = mode === 'edit' && skill?.origin === 'marketplace'

  // Load when opened
  useEffect(() => {
    if (!open) return
    setMode(initialMode)
    setFindings([])
    setAcknowledgeWarnings(false)
    setConfirmDelete(false)
    setAssignAgentId('')

    if (initialMode === 'new' || skillId === null) {
      setSkill(null)
      setName('')
      setDescription('')
      setCategory('')
      setTagsCsv('')
      setContent(SAMPLE_SKELETON)
      return
    }

    if (!workspace?.id) return
    setLoading(true)
    getWorkspaceSkillContent(workspace.id, skillId)
      .then((data) => {
        setSkill(data)
        setName(data.name)
        setDescription(data.description ?? '')
        setCategory(data.category ?? '')
        setTagsCsv((data.tags ?? []).join(', '))
        setContent(data.content ?? '')
      })
      .catch((err) => {
        toast.error('Failed to load skill', { description: err.message })
        onClose(false)
      })
      .finally(() => setLoading(false))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, initialMode, skillId, workspace?.id])

  const tagList = useMemo(
    () => tagsCsv.split(',').map((t) => t.trim()).filter(Boolean),
    [tagsCsv],
  )

  const tokenEstimate = useMemo(() => Math.ceil(content.length / 4), [content])

  const handleFilePick = useCallback((files: FileList | null) => {
    const file = files?.[0]
    if (!file) return
    if (file.size > MAX_UPLOAD_BYTES) {
      toast.error('File too large', { description: `Max ${MAX_UPLOAD_BYTES / 1024} KB. Got ${(file.size / 1024).toFixed(1)} KB.` })
      return
    }
    if (!/\.(md|markdown|txt)$/i.test(file.name)) {
      toast.error('Unsupported file type', { description: 'Only .md, .markdown, or .txt files are accepted.' })
      return
    }
    const reader = new FileReader()
    reader.onload = () => {
      const text = typeof reader.result === 'string' ? reader.result : ''
      setContent(text)
    }
    reader.onerror = () => {
      toast.error('Read failed', { description: 'Could not read the file.' })
    }
    reader.readAsText(file)
  }, [toast])

  const handleSave = useCallback(async () => {
    if (!workspace?.id) return
    setSaving(true)
    setFindings([])
    try {
      if (isNew) {
        if (!name.trim()) {
          toast.error('Name required')
          setSaving(false)
          return
        }
        const result = await createWorkspaceSkill(workspace.id, {
          name: name.trim(),
          content,
          description: description.trim() || undefined,
          category: category.trim() || undefined,
          tags: tagList.length ? tagList : undefined,
          acknowledge_warnings: acknowledgeWarnings,
        })
        setFindings(result.warnings ?? [])
        onClose(true)
        return
      }

      if (skillId === null) return
      const result = await updateWorkspaceSkill(workspace.id, skillId, {
        content,
        description: description.trim() || undefined,
        category: category.trim() || undefined,
        tags: tagList.length ? tagList : undefined,
        acknowledge_warnings: acknowledgeWarnings,
      })
      setFindings(result.warnings ?? [])
      onClose(true)
    } catch (err: any) {
      if (err instanceof SkillScanError) {
        setFindings(err.findings)
        if (err.status === 'warnings') {
          // High-severity — let the user toggle the acknowledge box and retry
          toast('Review warnings', { description: 'High-severity findings — confirm and save again to proceed.' })
        } else {
          toast.error('Skill blocked', { description: 'Critical security findings must be removed before saving.' })
        }
      }
      // Non-scan errors already toasted by the hook
    } finally {
      setSaving(false)
    }
  }, [workspace?.id, isNew, skillId, name, content, description, category, tagList, acknowledgeWarnings, createWorkspaceSkill, updateWorkspaceSkill, onClose, toast])

  const handleDelete = useCallback(async () => {
    if (!workspace?.id || skillId === null) return
    setSaving(true)
    const ok = await deleteWorkspaceSkill(workspace.id, skillId)
    setSaving(false)
    if (ok) onClose(true)
  }, [workspace?.id, skillId, deleteWorkspaceSkill, onClose])

  const handleAssign = useCallback(async () => {
    if (!assignAgentId || skillId === null) return
    const agentIdNum = parseInt(assignAgentId, 10)
    if (Number.isNaN(agentIdNum)) return
    const ok = await assignSkillsToAgent(agentIdNum, [skillId], false)
    if (ok) {
      setAssignAgentId('')
    }
  }, [assignAgentId, skillId, assignSkillsToAgent])

  const blocking = findings.filter((f) => f.severity === 'critical')
  const warnings = findings.filter((f) => f.severity === 'high')

  return (
    <>
      <Dialog open={open} onOpenChange={(o) => !o && onClose(false)}>
        <DialogContent className="max-w-4xl max-h-[92vh] overflow-hidden flex flex-col">
          <DialogHeader className="space-y-2">
            <div className="flex items-start justify-between gap-3">
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10 text-primary">
                  <Brain className="h-5 w-5" />
                </div>
                <div>
                  <DialogTitle className="text-xl">
                    {isNew ? 'New Skill' : (skill?.name || 'Skill')}
                  </DialogTitle>
                  <DialogDescription className="mt-0.5 text-xs flex flex-wrap items-center gap-1.5">
                    {isView && (
                      <Badge variant="outline" className="text-[10px]">
                        <Eye className="mr-1 h-3 w-3" /> View
                      </Badge>
                    )}
                    {!isView && !isNew && (
                      <Badge variant="outline" className="text-[10px]">
                        <Pencil className="mr-1 h-3 w-3" /> Edit
                      </Badge>
                    )}
                    {isNew && (
                      <Badge variant="outline" className="text-[10px]">
                        <Plus className="mr-1 h-3 w-3" /> New
                      </Badge>
                    )}
                    {skill?.origin === 'marketplace' && (
                      <Badge variant="secondary" className="text-[10px]">Marketplace</Badge>
                    )}
                    {skill?.origin === 'workspace' && skill?.forked_from_skill_id && (
                      <Badge variant="secondary" className="text-[10px]">
                        <GitBranch className="mr-1 h-3 w-3" />
                        Forked from #{skill.forked_from_skill_id}
                      </Badge>
                    )}
                    {willFork && (
                      <span className="text-amber-500">
                        Saving will fork this marketplace skill into your workspace.
                      </span>
                    )}
                  </DialogDescription>
                </div>
              </div>

              {!isNew && skill && !isView && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setMode('view')}
                >
                  <Eye className="mr-1.5 h-3.5 w-3.5" />
                  View
                </Button>
              )}
              {isView && skill && (
                <Button size="sm" onClick={() => setMode('edit')}>
                  <Pencil className="mr-1.5 h-3.5 w-3.5" />
                  {skill.origin === 'marketplace' ? 'Fork & edit' : 'Edit'}
                </Button>
              )}
            </div>
          </DialogHeader>

          {/* Body */}
          <div className="flex-1 overflow-y-auto space-y-4 pr-1">
            {loading ? (
              <div className="flex items-center justify-center py-16">
                <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
              </div>
            ) : (
              <>
                {/* Metadata fields (hidden in view mode for marketplace, shown for owned) */}
                {!isView && (
                  <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                    {isNew && (
                      <div className="sm:col-span-2">
                        <label className="text-xs font-medium text-muted-foreground">Name *</label>
                        <Input
                          value={name}
                          onChange={(e) => setName(e.target.value)}
                          placeholder="my-skill-slug"
                          className="mt-1"
                        />
                      </div>
                    )}
                    <div>
                      <label className="text-xs font-medium text-muted-foreground">Category</label>
                      <Input
                        value={category}
                        onChange={(e) => setCategory(e.target.value)}
                        placeholder="e.g. marketing"
                        className="mt-1"
                      />
                    </div>
                    <div>
                      <label className="text-xs font-medium text-muted-foreground">Tags (comma-separated)</label>
                      <Input
                        value={tagsCsv}
                        onChange={(e) => setTagsCsv(e.target.value)}
                        placeholder="e.g. social, copy, instagram"
                        className="mt-1"
                      />
                    </div>
                    <div className="sm:col-span-2">
                      <label className="text-xs font-medium text-muted-foreground">Description</label>
                      <Input
                        value={description}
                        onChange={(e) => setDescription(e.target.value)}
                        placeholder="One-line summary"
                        className="mt-1"
                      />
                    </div>
                  </div>
                )}

                {/* Findings panel */}
                {findings.length > 0 && (
                  <FindingsPanel
                    blocking={blocking}
                    warnings={warnings}
                    acknowledged={acknowledgeWarnings}
                    onAcknowledge={setAcknowledgeWarnings}
                    showAcknowledge={blocking.length === 0 && warnings.length > 0}
                  />
                )}

                {/* Editor / Viewer */}
                <div>
                  <div className="flex items-center justify-between mb-1.5">
                    <label className="text-xs font-medium text-muted-foreground">
                      SKILL.md content
                    </label>
                    <div className="flex items-center gap-3 text-xs text-muted-foreground">
                      <span className="tabular-nums">~{tokenEstimate.toLocaleString()} tokens</span>
                      {!isView && (
                        <button
                          type="button"
                          onClick={() => fileInputRef.current?.click()}
                          className="inline-flex items-center gap-1 hover:text-foreground"
                        >
                          <Upload className="h-3 w-3" />
                          Upload .md
                        </button>
                      )}
                      <input
                        ref={fileInputRef}
                        type="file"
                        accept=".md,.markdown,.txt,text/markdown,text/plain"
                        className="hidden"
                        onChange={(e) => {
                          handleFilePick(e.target.files)
                          e.target.value = ''
                        }}
                      />
                    </div>
                  </div>
                  <Textarea
                    value={content}
                    onChange={(e) => setContent(e.target.value)}
                    readOnly={isView}
                    className="font-mono text-sm min-h-[420px] resize-y"
                    placeholder="---&#10;name: my-skill&#10;description: …&#10;---&#10;&#10;# Skill body"
                  />
                </div>
              </>
            )}
          </div>

          {/* Footer */}
          <div className="border-t pt-3 mt-3 flex flex-wrap items-center gap-2">
            {!isNew && skill && (
              <div className="flex items-center gap-2">
                <Select value={assignAgentId} onValueChange={setAssignAgentId}>
                  <SelectTrigger className="w-[200px] h-9">
                    <SelectValue placeholder="Assign to agent…" />
                  </SelectTrigger>
                  <SelectContent>
                    {(agents as any[]).map((agent) => (
                      <SelectItem key={agent.id} value={String(agent.id)}>
                        {agent.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={handleAssign}
                  disabled={!assignAgentId}
                >
                  <UserPlus className="mr-1.5 h-3.5 w-3.5" />
                  Assign
                </Button>
              </div>
            )}

            <div className="ml-auto flex items-center gap-2">
              {skill?.origin === 'workspace' && !isNew && (
                <Button
                  variant="ghost"
                  size="sm"
                  className="text-destructive hover:text-destructive"
                  onClick={() => setConfirmDelete(true)}
                  disabled={saving}
                >
                  <Trash2 className="mr-1.5 h-3.5 w-3.5" />
                  Delete
                </Button>
              )}
              <Button variant="ghost" onClick={() => onClose(false)} disabled={saving}>
                {isView ? 'Close' : 'Cancel'}
              </Button>
              {!isView && (
                <Button onClick={handleSave} disabled={saving || (warnings.length > 0 && !acknowledgeWarnings && blocking.length === 0)}>
                  {saving ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Save className="mr-2 h-4 w-4" />}
                  {isNew ? 'Create' : willFork ? 'Fork & save' : 'Save'}
                </Button>
              )}
            </div>
          </div>
        </DialogContent>
      </Dialog>

      <AlertDialog open={confirmDelete} onOpenChange={setConfirmDelete}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete this skill?</AlertDialogTitle>
            <AlertDialogDescription>
              This permanently removes the skill from your workspace and unassigns it from any agents.
              {skill?.forked_from_skill_id && ' The original marketplace skill will still be available to re-install.'}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleDelete}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  )
}

function FindingsPanel({
  blocking,
  warnings,
  acknowledged,
  onAcknowledge,
  showAcknowledge,
}: {
  blocking: ScannerFinding[]
  warnings: ScannerFinding[]
  acknowledged: boolean
  onAcknowledge: (v: boolean) => void
  showAcknowledge: boolean
}) {
  return (
    <div className="space-y-2">
      {blocking.length > 0 && (
        <div className="rounded-md border border-destructive/40 bg-destructive/5 p-3">
          <div className="flex items-center gap-2 text-destructive font-medium text-sm">
            <ShieldAlert className="h-4 w-4" />
            {blocking.length} critical issue{blocking.length === 1 ? '' : 's'} — save blocked
          </div>
          <ul className="mt-2 space-y-1 text-xs text-destructive/90">
            {blocking.map((f, i) => (
              <li key={i} className="font-mono">
                <span className="text-destructive">L{f.line}</span> · {f.description}
                {f.matched_text && <span className="opacity-70"> — “{f.matched_text}”</span>}
              </li>
            ))}
          </ul>
        </div>
      )}
      {warnings.length > 0 && (
        <div className="rounded-md border border-amber-500/40 bg-amber-500/5 p-3">
          <div className="flex items-center gap-2 text-amber-500 font-medium text-sm">
            <AlertTriangle className="h-4 w-4" />
            {warnings.length} warning{warnings.length === 1 ? '' : 's'}
          </div>
          <ul className="mt-2 space-y-1 text-xs text-amber-200">
            {warnings.map((f, i) => (
              <li key={i} className="font-mono">
                <span className="text-amber-400">L{f.line}</span> · {f.description}
                {f.matched_text && <span className="opacity-70"> — “{f.matched_text}”</span>}
              </li>
            ))}
          </ul>
          {showAcknowledge && (
            <label className="mt-3 flex items-start gap-2 text-xs cursor-pointer">
              <Checkbox
                checked={acknowledged}
                onCheckedChange={(checked) => onAcknowledge(checked === true)}
                className="mt-0.5"
              />
              <span>
                I&rsquo;ve reviewed these warnings and want to save anyway. (You can resolve them later — they don&rsquo;t block agent execution.)
              </span>
            </label>
          )}
        </div>
      )}
    </div>
  )
}
