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
  active: 'bg-emerald-500',
  inactive: 'bg-zinc-500',
  training: 'bg-amber-500',
}

const TEAM_COLORS: Record<string, string> = {
  engineering: 'border-blue-500/60',
  marketing: 'border-purple-500/60',
  sales: 'border-green-500/60',
  content: 'border-orange-500/60',
  finance: 'border-yellow-500/60',
  operations: 'border-cyan-500/60',
  support: 'border-pink-500/60',
  research: 'border-indigo-500/60',
  hr: 'border-rose-500/60',
}

function getTeamBorder(team: string | null): string {
  if (!team) return 'border-zinc-700/60'
  return TEAM_COLORS[team.toLowerCase()] ?? 'border-zinc-600/60'
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
  const teamBorder = getTeamBorder(data.team)
  const modelShort = getModelShort(data.model)

  return (
    <div
      className={cn(
        'rounded-xl border-2 bg-zinc-900/90 backdrop-blur-sm px-4 py-3 w-[220px]',
        'transition-all duration-200 hover:shadow-lg hover:shadow-orange-500/10',
        teamBorder,
        data.isSystemAgent && 'ring-2 ring-orange-500/40 border-orange-500/60',
      )}
    >
      {/* Target handle (from parent) */}
      <Handle
        type="target"
        position={Position.Top}
        className="!bg-zinc-600 !border-zinc-500 !w-2 !h-2"
      />

      {/* Header: status dot + name */}
      <div className="flex items-center gap-2 mb-1.5">
        <div className={cn('w-2 h-2 rounded-full shrink-0', STATUS_COLORS[data.status] ?? 'bg-zinc-500')} />
        <span className="text-sm font-semibold text-zinc-100 truncate">
          {data.name}
        </span>
        {data.isSystemAgent && (
          <Shield className="w-3 h-3 text-orange-400 shrink-0" />
        )}
      </div>

      {/* Job title */}
      {data.jobTitle && (
        <div className="text-xs text-zinc-400 truncate mb-1.5">
          {data.jobTitle}
        </div>
      )}

      {/* Team badge */}
      {data.team && (
        <div className="mb-2">
          <span className={cn(
            'inline-block text-[10px] font-medium px-1.5 py-0.5 rounded-full',
            'bg-zinc-800 text-zinc-300 border border-zinc-700/50'
          )}>
            {data.team}
          </span>
        </div>
      )}

      {/* Stats row */}
      <div className="flex items-center gap-3 text-[10px] text-zinc-500">
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
        className="!bg-zinc-600 !border-zinc-500 !w-2 !h-2"
      />
    </div>
  )
}

export const OrgChartNode = memo(OrgChartNodeInner)
