/**
 * Workspace Skills Tab — user-local skills library on the Agents page.
 *
 * Shows the workspace's enabled marketplace skills + workspace-owned (forked or
 * user-created) skills. Click a card to view; edit forks marketplace skills on
 * first save. "+ New Skill" opens a paste/upload editor.
 */

'use client'

import { useCallback, useEffect, useMemo, useState } from 'react'
import { Brain, Plus, Search, Pencil, Eye, GitBranch, ShoppingBag } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { useToast } from '@/hooks/use-toast'
import { useWorkspace } from '@/components/workspace-provider'
import { useSkillsApi } from '@/hooks/use-skills-api'

import { SkillEditorModal, type SkillEditorMode } from './skill-editor-modal'

interface WorkspaceSkillRow {
  skill_id: number
  name: string
  description: string | null
  category: string | null
  skill_version: string | null
  tags: string[] | null
  estimated_tokens: number
  skill_source: string | null
  enabled_at: string | null
  origin: 'marketplace' | 'workspace'
  forked_from_skill_id: number | null
}

export function WorkspaceSkillsTab() {
  const { workspace } = useWorkspace()
  const { toast } = useToast()
  const { listWorkspaceSkills } = useSkillsApi()

  const [skills, setSkills] = useState<WorkspaceSkillRow[]>([])
  const [search, setSearch] = useState('')
  const [originFilter, setOriginFilter] = useState<'all' | 'workspace' | 'marketplace'>('all')
  const [loading, setLoading] = useState(true)

  const [editorOpen, setEditorOpen] = useState(false)
  const [editorMode, setEditorMode] = useState<SkillEditorMode>('view')
  const [editorSkillId, setEditorSkillId] = useState<number | null>(null)

  const loadSkills = useCallback(async () => {
    if (!workspace?.id) return
    setLoading(true)
    try {
      const items = await listWorkspaceSkills(workspace.id)
      setSkills(items as WorkspaceSkillRow[])
    } catch (err: any) {
      toast({ title: 'Failed to load skills', description: err.message, variant: 'destructive' })
    } finally {
      setLoading(false)
    }
  }, [workspace?.id, listWorkspaceSkills, toast])

  useEffect(() => {
    loadSkills()
  }, [loadSkills])

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase()
    return skills.filter((s) => {
      if (originFilter !== 'all' && s.origin !== originFilter) return false
      if (!q) return true
      return (
        s.name.toLowerCase().includes(q) ||
        (s.description || '').toLowerCase().includes(q) ||
        (s.tags || []).some((t) => t.toLowerCase().includes(q))
      )
    })
  }, [skills, search, originFilter])

  const openViewer = useCallback((skillId: number, mode: SkillEditorMode) => {
    setEditorSkillId(skillId)
    setEditorMode(mode)
    setEditorOpen(true)
  }, [])

  const openNew = useCallback(() => {
    setEditorSkillId(null)
    setEditorMode('new')
    setEditorOpen(true)
  }, [])

  const handleEditorClose = useCallback((didChange: boolean) => {
    setEditorOpen(false)
    if (didChange) {
      loadSkills()
    }
  }, [loadSkills])

  const ownedCount = useMemo(() => skills.filter((s) => s.origin === 'workspace').length, [skills])
  const marketplaceCount = useMemo(() => skills.filter((s) => s.origin === 'marketplace').length, [skills])

  return (
    <div className="space-y-4">
      {/* Header row */}
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
        <div className="relative flex-1 min-w-[220px] max-w-md">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search skills…"
            className="pl-9"
          />
        </div>

        <div className="flex gap-2">
          <Button
            variant={originFilter === 'all' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setOriginFilter('all')}
          >
            All <span className="ml-1.5 text-xs text-muted-foreground">{skills.length}</span>
          </Button>
          <Button
            variant={originFilter === 'workspace' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setOriginFilter('workspace')}
          >
            <GitBranch className="mr-1.5 h-3.5 w-3.5" />
            Yours <span className="ml-1.5 text-xs text-muted-foreground">{ownedCount}</span>
          </Button>
          <Button
            variant={originFilter === 'marketplace' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setOriginFilter('marketplace')}
          >
            <ShoppingBag className="mr-1.5 h-3.5 w-3.5" />
            Marketplace <span className="ml-1.5 text-xs text-muted-foreground">{marketplaceCount}</span>
          </Button>
        </div>

        <div className="ml-auto">
          <Button onClick={openNew}>
            <Plus className="mr-2 h-4 w-4" />
            New Skill
          </Button>
        </div>
      </div>

      {/* Cards */}
      {loading ? (
        <SkillCardsSkeleton />
      ) : filtered.length === 0 ? (
        <EmptyState
          hasAny={skills.length > 0}
          onCreate={openNew}
          searching={search.trim().length > 0 || originFilter !== 'all'}
        />
      ) : (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {filtered.map((skill) => (
            <SkillCard
              key={skill.skill_id}
              skill={skill}
              onView={() => openViewer(skill.skill_id, 'view')}
              onEdit={() => openViewer(skill.skill_id, 'edit')}
            />
          ))}
        </div>
      )}

      <SkillEditorModal
        open={editorOpen}
        mode={editorMode}
        skillId={editorSkillId}
        onClose={handleEditorClose}
      />
    </div>
  )
}

function SkillCard({
  skill,
  onView,
  onEdit,
}: {
  skill: WorkspaceSkillRow
  onView: () => void
  onEdit: () => void
}) {
  const isWorkspace = skill.origin === 'workspace'
  return (
    <Card className="glass-card group transition-all hover:border-primary/40">
      <CardContent className="p-5 space-y-3">
        <div className="flex items-start justify-between gap-3">
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary">
            <Brain className="h-5 w-5" />
          </div>
          <div className="flex flex-wrap justify-end gap-1.5">
            {isWorkspace ? (
              <Badge variant="secondary" className="text-[10px]">
                <GitBranch className="mr-1 h-3 w-3" />
                {skill.forked_from_skill_id ? 'Forked' : 'Yours'}
              </Badge>
            ) : (
              <Badge variant="outline" className="text-[10px]">
                <ShoppingBag className="mr-1 h-3 w-3" />
                Marketplace
              </Badge>
            )}
            {skill.skill_version && (
              <Badge variant="outline" className="text-[10px]">v{skill.skill_version}</Badge>
            )}
          </div>
        </div>

        <div>
          <h3 className="font-semibold text-base line-clamp-1">{skill.name}</h3>
          {skill.description && (
            <p className="mt-1 text-sm text-muted-foreground line-clamp-2">{skill.description}</p>
          )}
        </div>

        <div className="flex items-center justify-between text-xs text-muted-foreground">
          <span>{skill.category ?? 'Uncategorised'}</span>
          <span className="tabular-nums">~{skill.estimated_tokens.toLocaleString()} tok</span>
        </div>

        <div className="flex gap-2 pt-1">
          <Button variant="outline" size="sm" className="flex-1" onClick={onView}>
            <Eye className="mr-1.5 h-3.5 w-3.5" />
            View
          </Button>
          <Button variant="outline" size="sm" className="flex-1" onClick={onEdit}>
            <Pencil className="mr-1.5 h-3.5 w-3.5" />
            {isWorkspace ? 'Edit' : 'Fork & edit'}
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}

function SkillCardsSkeleton() {
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
      {[0, 1, 2, 3, 4, 5].map((i) => (
        <Card key={i} className="glass-card">
          <CardContent className="p-5 space-y-3">
            <div className="h-10 w-10 rounded-lg bg-secondary/50" />
            <div className="h-4 w-2/3 rounded bg-secondary/50" />
            <div className="h-3 w-full rounded bg-secondary/30" />
            <div className="h-3 w-4/5 rounded bg-secondary/30" />
          </CardContent>
        </Card>
      ))}
    </div>
  )
}

function EmptyState({
  hasAny,
  onCreate,
  searching,
}: {
  hasAny: boolean
  onCreate: () => void
  searching: boolean
}) {
  if (searching) {
    return (
      <div className="flex flex-col items-center justify-center gap-3 py-16 text-center text-muted-foreground">
        <Search className="h-10 w-10" strokeWidth={1.5} />
        <div className="text-sm">No skills match your filters.</div>
      </div>
    )
  }
  return (
    <div className="flex flex-col items-center justify-center gap-3 py-16 text-center">
      <Brain className="h-10 w-10 text-muted-foreground" strokeWidth={1.5} />
      <div className="text-base font-medium">
        {hasAny ? 'No skills in this view' : 'No skills yet'}
      </div>
      <div className="max-w-sm text-sm text-muted-foreground">
        Skills give your agents repeatable how-tos. Install one from the Marketplace, or paste a SKILL.md to author your own.
      </div>
      <Button onClick={onCreate} className="mt-2">
        <Plus className="mr-2 h-4 w-4" />
        New Skill
      </Button>
    </div>
  )
}
