'use client'

import { useQuery } from '@tanstack/react-query'
import { Shield, ShieldAlert, ShieldCheck, ShieldX, Loader2 } from 'lucide-react'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { apiClient } from '@/lib/api-client'
import { cn } from '@/lib/utils'

interface ValidationResult {
  success: boolean
  pass: boolean
  failures: string[]
  warnings: string[]
  agent_name: string
  blueprint_name: string | null
}

interface AgentReadinessBadgeProps {
  agentId: number
  className?: string
}

export function AgentReadinessBadge({ agentId, className }: AgentReadinessBadgeProps) {
  const { data, isLoading } = useQuery<ValidationResult>({
    queryKey: ['agent-readiness', agentId],
    queryFn: () =>
      apiClient.request<ValidationResult>(`/api/agents/${agentId}/validate`),
    staleTime: 60_000,
    retry: false,
  })

  if (isLoading) {
    return <Loader2 className={cn('w-4 h-4 animate-spin text-muted-foreground', className)} />
  }

  // No blueprint configured or API not available yet
  if (!data?.success || data.blueprint_name === null) {
    return (
      <Tooltip>
        <TooltipTrigger asChild>
          <Shield className={cn('w-4 h-4 text-muted-foreground/50', className)} />
        </TooltipTrigger>
        <TooltipContent side="top" className="text-xs max-w-[200px]">
          No governance blueprint configured
        </TooltipContent>
      </Tooltip>
    )
  }

  const passed = data.pass
  const hasWarnings = data.warnings.length > 0
  const failures = data.failures
  const warnings = data.warnings

  // Determine icon and color
  let Icon = ShieldCheck
  let colorClass = 'text-emerald-400'
  let label = 'Ready'

  if (!passed) {
    Icon = ShieldX
    colorClass = 'text-red-400'
    label = 'Not Ready'
  } else if (hasWarnings) {
    Icon = ShieldAlert
    colorClass = 'text-amber-400'
    label = 'Warnings'
  }

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Icon className={cn('w-4 h-4', colorClass, className)} />
      </TooltipTrigger>
      <TooltipContent side="top" className="text-xs max-w-[260px] space-y-1">
        <p className="font-medium">{label} — {data.blueprint_name}</p>
        {failures.map((f, i) => (
          <p key={i} className="text-red-400">✗ {f}</p>
        ))}
        {warnings.map((w, i) => (
          <p key={i} className="text-amber-400">⚠ {w}</p>
        ))}
        {passed && !hasWarnings && (
          <p className="text-emerald-400">All checks passed</p>
        )}
      </TooltipContent>
    </Tooltip>
  )
}
