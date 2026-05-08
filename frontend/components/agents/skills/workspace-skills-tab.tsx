/**
 * Workspace Skills Tab — user-local skills library on the Agents page.
 *
 * Shows the workspace's enabled marketplace skills + workspace-owned (forked or
 * user-created) skills. Click a card to view; edit forks marketplace skills on
 * first save. "+ New Skill" opens a paste/upload editor.
 */

'use client'

import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Brain,
  Eye,
  GitBranch,
  Pencil,
  Plus,
  Search,
  ShoppingBag,
  Trash2,
} from 'lucide-react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
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
import { useToast } from '@/hooks/use-toast'
import { useWorkspace } from '@/components/workspace-provider'
import { useSkillsApi } from '@/hooks/use-skills-api'
import { useSystemIcons } from '@/hooks/use-system-config-api'
import { PremiumIcon } from '@/components/shared'
import type { ViewMode } from '@/components/shared/view-toggle'

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
  assigned_agent_count: number
}

interface Props {
  viewMode?: ViewMode
}

export function WorkspaceSkillsTab({ viewMode = 'grid' }: Props) {
  const { workspace } = useWorkspace()
  const { toast } = useToast()
  const { listWorkspaceSkills, deleteWorkspaceSkill, disableWorkspaceSkill } = useSkillsApi()
  const { data: iconMappings = {} } = useSystemIcons()

  const resolveIconName = useCallback(
    (skill: WorkspaceSkillRow): string | null =>
      (skill.category && iconMappings[skill.category]) || iconMappings['global_skill'] || null,
    [iconMappings],
  )

  const [skills, setSkills] = useState<WorkspaceSkillRow[]>([])
  const [search, setSearch] = useState('')
  const [originFilter, setOriginFilter] = useState<'all' | 'workspace' | 'marketplace'>('all')
  const [assignmentFilter, setAssignmentFilter] = useState<'all' | 'assigned' | 'unassigned'>('all')
  const [loading, setLoading] = useState(true)

  const [editorOpen, setEditorOpen] = useState(false)
  const [editorMode, setEditorMode] = useState<SkillEditorMode>('view')
  const [editorSkillId, setEditorSkillId] = useState<number | null>(null)

  const [removeTarget, setRemoveTarget] = useState<WorkspaceSkillRow | null>(null)
  const [removing, setRemoving] = useState(false)

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
      if (assignmentFilter === 'assigned' && s.assigned_agent_count === 0) return false
      if (assignmentFilter === 'unassigned' && s.assigned_agent_count > 0) return false
      if (!q) return true
      return (
        s.name.toLowerCase().includes(q) ||
        (s.description || '').toLowerCase().includes(q) ||
        (s.tags || []).some((t) => t.toLowerCase().includes(q))
      )
    })
  }, [skills, search, originFilter, assignmentFilter])

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

  const confirmRemove = useCallback(async () => {
    if (!removeTarget || !workspace?.id) return
    setRemoving(true)
    // Workspace-owned skills are deleted entirely; marketplace skills are
    // disabled (junction dropped — original stays in the catalogue).
    const ok = removeTarget.origin === 'workspace'
      ? await deleteWorkspaceSkill(workspace.id, removeTarget.skill_id)
      : await disableWorkspaceSkill(workspace.id, removeTarget.skill_id)
    setRemoving(false)
    if (ok) {
      setRemoveTarget(null)
      loadSkills()
    }
  }, [removeTarget, workspace?.id, deleteWorkspaceSkill, disableWorkspaceSkill, loadSkills])

  const ownedCount = useMemo(() => skills.filter((s) => s.origin === 'workspace').length, [skills])
  const marketplaceCount = useMemo(() => skills.filter((s) => s.origin === 'marketplace').length, [skills])
  const assignedCount = useMemo(() => skills.filter((s) => s.assigned_agent_count > 0).length, [skills])
  const unassignedCount = useMemo(() => skills.filter((s) => s.assigned_agent_count === 0).length, [skills])

  return (
    <div className="space-y-4">
      {/* Header row: search + new */}
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

        <div className="ml-auto">
          <Button onClick={openNew}>
            <Plus className="mr-2 h-4 w-4" />
            New Skill
          </Button>
        </div>
      </div>

      {/* Filter row */}
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-xs uppercase tracking-wide text-muted-foreground mr-1">Origin</span>
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

        <span className="ml-3 text-xs uppercase tracking-wide text-muted-foreground mr-1">Assignment</span>
        <Button
          variant={assignmentFilter === 'all' ? 'default' : 'outline'}
          size="sm"
          onClick={() => setAssignmentFilter('all')}
        >
          All
        </Button>
        <Button
          variant={assignmentFilter === 'assigned' ? 'default' : 'outline'}
          size="sm"
          onClick={() => setAssignmentFilter('assigned')}
        >
          Assigned <span className="ml-1.5 text-xs text-muted-foreground">{assignedCount}</span>
        </Button>
        <Button
          variant={assignmentFilter === 'unassigned' ? 'default' : 'outline'}
          size="sm"
          onClick={() => setAssignmentFilter('unassigned')}
        >
          Unassigned <span className="ml-1.5 text-xs text-muted-foreground">{unassignedCount}</span>
        </Button>
      </div>

      {/* Cards or list */}
      {loading ? (
        <SkillCardsSkeleton viewMode={viewMode} />
      ) : filtered.length === 0 ? (
        <EmptyState
          hasAny={skills.length > 0}
          onCreate={openNew}
          searching={
            search.trim().length > 0 ||
            originFilter !== 'all' ||
            assignmentFilter !== 'all'
          }
        />
      ) : viewMode === 'list' ? (
        <div className="space-y-2">
          {filtered.map((skill) => (
            <SkillRow
              key={skill.skill_id}
              skill={skill}
              iconName={resolveIconName(skill)}
              onView={() => openViewer(skill.skill_id, 'view')}
              onEdit={() => openViewer(skill.skill_id, 'edit')}
              onRemove={() => setRemoveTarget(skill)}
            />
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {filtered.map((skill) => (
            <SkillCard
              key={skill.skill_id}
              skill={skill}
              iconName={resolveIconName(skill)}
              onView={() => openViewer(skill.skill_id, 'view')}
              onEdit={() => openViewer(skill.skill_id, 'edit')}
              onRemove={() => setRemoveTarget(skill)}
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

      <AlertDialog open={!!removeTarget} onOpenChange={(open) => !open && setRemoveTarget(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              {removeTarget?.origin === 'workspace' ? 'Delete from workspace?' : 'Remove from workspace?'}
            </AlertDialogTitle>
            <AlertDialogDescription>
              {removeTarget?.origin === 'workspace'
                ? removeTarget?.forked_from_skill_id
                    ? `Deletes your forked copy of "${removeTarget?.name}" from this workspace. The original marketplace skill stays available to re-install. This cannot be undone.`
                    : `Deletes "${removeTarget?.name}" from this workspace permanently. Nothing in the marketplace or other workspaces is affected. This cannot be undone.`
                : `Removes "${removeTarget?.name}" from this workspace. The marketplace original stays in the catalogue and can be re-installed any time. Other workspaces are unaffected.`}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={removing}>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={confirmRemove}
              disabled={removing}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              {removeTarget?.origin === 'workspace' ? 'Delete' : 'Remove'}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  )
}

function SkillCard({
  skill,
  iconName,
  onView,
  onEdit,
  onRemove,
}: {
  skill: WorkspaceSkillRow
  iconName: string | null
  onView: () => void
  onEdit: () => void
  onRemove: () => void
}) {
  const isWorkspace = skill.origin === 'workspace'
  const canRemove = skill.assigned_agent_count === 0
  const removeLabel = isWorkspace ? 'Delete from workspace' : 'Remove from workspace'
  return (
    <Card className="glass-card group transition-all hover:border-primary/40">
      <CardContent className="p-5 space-y-3">
        <div className="flex items-start justify-between gap-3">
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary">
            {iconName ? (
              <PremiumIcon name={iconName} size={28} className="text-primary" />
            ) : (
              <Brain className="h-5 w-5" />
            )}
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

        <div className="flex items-center justify-between text-xs">
          <span className={skill.assigned_agent_count > 0 ? 'text-foreground' : 'text-muted-foreground'}>
            {skill.assigned_agent_count > 0
              ? `Assigned to ${skill.assigned_agent_count} ${skill.assigned_agent_count === 1 ? 'agent' : 'agents'}`
              : 'Unassigned'}
          </span>
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
          {canRemove && (
            <Button
              variant="outline"
              size="icon"
              className="h-9 w-9 shrink-0 text-destructive hover:text-destructive hover:border-destructive/50"
              onClick={onRemove}
              aria-label={removeLabel}
              title={removeLabel}
            >
              <Trash2 className="h-3.5 w-3.5" />
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  )
}

function SkillRow({
  skill,
  iconName,
  onView,
  onEdit,
  onRemove,
}: {
  skill: WorkspaceSkillRow
  iconName: string | null
  onView: () => void
  onEdit: () => void
  onRemove: () => void
}) {
  const isWorkspace = skill.origin === 'workspace'
  const canRemove = skill.assigned_agent_count === 0
  const removeLabel = isWorkspace ? 'Delete from workspace' : 'Remove from workspace'
  return (
    <div className="glass-card group flex items-center gap-4 p-3 transition-colors hover:border-primary/40">
      <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md bg-primary/10 text-primary">
        {iconName ? (
          <PremiumIcon name={iconName} size={22} className="text-primary" />
        ) : (
          <Brain className="h-4 w-4" />
        )}
      </div>
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span className="font-medium truncate">{skill.name}</span>
          {isWorkspace ? (
            <Badge variant="secondary" className="text-[10px] shrink-0">
              <GitBranch className="mr-1 h-3 w-3" />
              {skill.forked_from_skill_id ? 'Forked' : 'Yours'}
            </Badge>
          ) : (
            <Badge variant="outline" className="text-[10px] shrink-0">
              <ShoppingBag className="mr-1 h-3 w-3" />
              Marketplace
            </Badge>
          )}
          {skill.skill_version && (
            <Badge variant="outline" className="text-[10px] shrink-0">v{skill.skill_version}</Badge>
          )}
        </div>
        {skill.description && (
          <p className="text-xs text-muted-foreground truncate mt-0.5">{skill.description}</p>
        )}
      </div>
      <div className="hidden shrink-0 text-xs sm:block">
        <span className={skill.assigned_agent_count > 0 ? 'text-foreground' : 'text-muted-foreground'}>
          {skill.assigned_agent_count > 0
            ? `${skill.assigned_agent_count} ${skill.assigned_agent_count === 1 ? 'agent' : 'agents'}`
            : 'Unassigned'}
        </span>
      </div>
      <div className="hidden shrink-0 text-xs text-muted-foreground sm:block">
        {skill.category ?? 'Uncategorised'}
      </div>
      <div className="hidden shrink-0 text-xs text-muted-foreground tabular-nums md:block">
        ~{skill.estimated_tokens.toLocaleString()} tok
      </div>
      <div className="flex shrink-0 gap-1">
        <Button variant="ghost" size="sm" onClick={onView} aria-label="View">
          <Eye className="h-4 w-4" />
        </Button>
        <Button variant="ghost" size="sm" onClick={onEdit} aria-label={isWorkspace ? 'Edit' : 'Fork & edit'}>
          <Pencil className="h-4 w-4" />
        </Button>
        {canRemove && (
          <Button
            variant="ghost"
            size="sm"
            onClick={onRemove}
            aria-label={removeLabel}
            title={removeLabel}
            className="text-destructive hover:text-destructive"
          >
            <Trash2 className="h-4 w-4" />
          </Button>
        )}
      </div>
    </div>
  )
}

function SkillCardsSkeleton({ viewMode }: { viewMode: ViewMode }) {
  if (viewMode === 'list') {
    return (
      <div className="space-y-2">
        {[0, 1, 2, 3, 4].map((i) => (
          <div key={i} className="glass-card flex items-center gap-4 p-3">
            <div className="h-9 w-9 rounded-md bg-secondary/50" />
            <div className="flex-1 space-y-2">
              <div className="h-3 w-1/3 rounded bg-secondary/50" />
              <div className="h-2 w-2/3 rounded bg-secondary/30" />
            </div>
          </div>
        ))}
      </div>
    )
  }
  return (
    <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
      {[0, 1, 2, 3, 4, 5, 6, 7].map((i) => (
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
