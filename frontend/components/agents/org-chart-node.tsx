'use client'

import { memo } from 'react'
import { Handle, Position, type NodeProps } from 'reactflow'
import { Bot, Users, Cpu, Shield, Briefcase } from 'lucide-react'
import { cn } from '@/lib/utils'

export interface OrgChartNodeData {
  id: number
  name: string
  jobTitle: string | null
  team: string | null
  status: string
  model: string | null
  skills: string[]
  toolsCount: number
  directReportsCount: number
  isSystemAgent: boolean
}

const STATUS_COLORS: Record<string, string> = {
  active: 'bg-success',
  inactive: 'bg-muted-foreground',
  training: 'bg-warning',
}

// Palette of border + matching badge styles. Stable per team name via hash —
// any team string deterministically picks the same colour every render.
const TEAM_PALETTE: Array<{ border: string; badge: string }> = [
  { border: 'border-blue-500/60',    badge: 'bg-blue-500/15 text-blue-300 border-blue-500/30' },
  { border: 'border-green-500/60',   badge: 'bg-green-500/15 text-green-300 border-green-500/30' },
  { border: 'border-purple-500/60',  badge: 'bg-purple-500/15 text-purple-300 border-purple-500/30' },
  { border: 'border-pink-500/60',    badge: 'bg-pink-500/15 text-pink-300 border-pink-500/30' },
  { border: 'border-orange-500/60',  badge: 'bg-orange-500/15 text-orange-300 border-orange-500/30' },
  { border: 'border-yellow-500/60',  badge: 'bg-yellow-500/15 text-yellow-300 border-yellow-500/30' },
  { border: 'border-cyan-500/60',    badge: 'bg-cyan-500/15 text-cyan-300 border-cyan-500/30' },
  { border: 'border-indigo-500/60',  badge: 'bg-indigo-500/15 text-indigo-300 border-indigo-500/30' },
  { border: 'border-rose-500/60',    badge: 'bg-rose-500/15 text-rose-300 border-rose-500/30' },
  { border: 'border-teal-500/60',    badge: 'bg-teal-500/15 text-teal-300 border-teal-500/30' },
  { border: 'border-emerald-500/60', badge: 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30' },
  { border: 'border-amber-500/60',   badge: 'bg-amber-500/15 text-amber-300 border-amber-500/30' },
  { border: 'border-violet-500/60',  badge: 'bg-violet-500/15 text-violet-300 border-violet-500/30' },
  { border: 'border-fuchsia-500/60', badge: 'bg-fuchsia-500/15 text-fuchsia-300 border-fuchsia-500/30' },
  { border: 'border-sky-500/60',     badge: 'bg-sky-500/15 text-sky-300 border-sky-500/30' },
  { border: 'border-lime-500/60',    badge: 'bg-lime-500/15 text-lime-300 border-lime-500/30' },
]

function hashString(s: string): number {
  let h = 0
  for (let i = 0; i < s.length; i++) {
    h = (h * 31 + s.charCodeAt(i)) | 0
  }
  return Math.abs(h)
}

function getTeamStyle(team: string | null): { border: string; badge: string } {
  if (!team) {
    return {
      border: 'border-border',
      badge: 'bg-card text-foreground border-border',
    }
  }
  return TEAM_PALETTE[hashString(team.toLowerCase()) % TEAM_PALETTE.length]
}

function getModelShort(model: string | null): string {
  if (!model) return ''
  // "anthropic/claude-sonnet-4.6" → "sonnet-4.6"
  const parts = model.split('/')
  const name = parts[parts.length - 1]
  // Shorten common prefixes
  return name
    .replace('claude-', '')
    .replace('meta-llama-', '')
    .replace('mistral-', 'mis-')
    .replace('-instruct', '')
    .slice(0, 20)
}

function OrgChartNodeInner({ data }: NodeProps<OrgChartNodeData>) {
  const teamStyle = getTeamStyle(data.team)
  const modelShort = getModelShort(data.model)

  return (
    <div
      className={cn(
        'rounded-xl border-2 bg-card backdrop-blur-sm px-4 py-3 w-[220px]',
        'transition-all duration-220 hover:border-primary/20',
        teamStyle.border,
        data.isSystemAgent && 'ring-2 ring-primary/40 border-primary/60',
      )}
    >
      {/* Target handle (from parent) */}
      <Handle
        type="target"
        position={Position.Top}
        className="!bg-muted-foreground !border-border !w-2 !h-2"
      />

      {/* Header: status dot + name */}
      <div className="flex items-center gap-2 mb-1.5">
        <div className={cn('w-2 h-2 rounded-full shrink-0', STATUS_COLORS[data.status] ?? 'bg-muted-foreground')} />
        <span className="text-sm font-semibold text-foreground truncate">
          {data.name}
        </span>
        {data.isSystemAgent && (
          <Shield className="w-3 h-3 text-primary shrink-0" />
        )}
      </div>

      {/* Job title */}
      {data.jobTitle && (
        <div className="text-xs text-muted-foreground truncate mb-1.5">
          {data.jobTitle}
        </div>
      )}

      {/* Team badge */}
      {data.team && (
        <div className="mb-2">
          <span className={cn(
            'inline-block text-[10px] font-medium px-1.5 py-0.5 rounded-full border',
            teamStyle.badge,
          )}>
            {data.team}
          </span>
        </div>
      )}

      {/* Stats row */}
      <div className="flex items-center gap-3 text-[10px] text-muted-foreground">
        {modelShort && (
          <span className="flex items-center gap-1">
            <Cpu className="w-3 h-3" />
            {modelShort}
          </span>
        )}
        {data.toolsCount > 0 && (
          <span className="flex items-center gap-1">
            <Briefcase className="w-3 h-3" />
            {data.toolsCount}
          </span>
        )}
        {data.directReportsCount > 0 && (
          <span className="flex items-center gap-1">
            <Users className="w-3 h-3" />
            {data.directReportsCount}
          </span>
        )}
      </div>

      {/* Source handle (to children) */}
      <Handle
        type="source"
        position={Position.Bottom}
        className="!bg-muted-foreground !border-border !w-2 !h-2"
      />
    </div>
  )
}

export const OrgChartNode = memo(OrgChartNodeInner)
