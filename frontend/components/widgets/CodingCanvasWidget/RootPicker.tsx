'use client'

import { FolderOpen } from 'lucide-react'
import { Select, SelectContent, SelectGroup, SelectItem, SelectLabel, SelectTrigger, SelectValue } from '@/components/ui/select'
import { useRepoRoots } from './useRepoRoots'
import { WORKSPACE_ROOT } from './code-root'

/** PRD-235 W2 — choose the folder the Canvas (tree, terminal, and the chat's scope) is rooted at. */
export function RootPicker({ workspaceId, value, onChange }: { workspaceId: string; value: string; onChange: (root: string) => void }) {
  const { data } = useRepoRoots(workspaceId)
  const options = data ?? [{ value: WORKSPACE_ROOT, label: 'Workspace root', group: 'workspace' as const }]
  const known = options.some((o) => o.value === value)
  const projects = options.filter((o) => o.group === 'projects')
  const sessions = options.filter((o) => o.group === 'sessions')
  return (
    <div className="flex items-center gap-2 px-3 py-1.5 border-b border-border/20 text-xs" data-testid="code-root-picker">
      <FolderOpen className="h-3.5 w-3.5 text-muted-foreground" />
      <Select value={known ? value : value} onValueChange={onChange}>
        <SelectTrigger className="h-7 w-auto min-w-[220px] text-xs" aria-label="Code root">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value={WORKSPACE_ROOT}>Workspace root</SelectItem>
          {!known && <SelectItem value={value}>{value}</SelectItem>}
          {projects.length > 0 && (
            <SelectGroup>
              <SelectLabel>Your repositories (projects/)</SelectLabel>
              {projects.map((o) => <SelectItem key={o.value} value={o.value}>{o.label}</SelectItem>)}
            </SelectGroup>
          )}
          {sessions.length > 0 && (
            <SelectGroup>
              <SelectLabel>Claude Code sessions (sessions/)</SelectLabel>
              {sessions.map((o) => <SelectItem key={o.value} value={o.value}>{o.label} · {o.value}</SelectItem>)}
            </SelectGroup>
          )}
        </SelectContent>
      </Select>
      <span className="text-muted-foreground">Auto works in this folder while Code mode is open.</span>
    </div>
  )
}
