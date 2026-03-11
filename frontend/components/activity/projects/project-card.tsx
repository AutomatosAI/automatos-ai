'use client'

import { formatDistanceToNow } from 'date-fns'
import { Bot, CheckCircle2 } from 'lucide-react'
import { motion, useReducedMotion } from 'framer-motion'
import { PremiumIcon } from '@/components/shared'
import { Badge } from '@/components/ui/badge'
import { cn } from '@/lib/utils'
import type { Project } from '@/hooks/use-projects'

// ─── Config Maps ────────────────────────────────────────

const STATUS_CONFIG: Record<Project['status'], { label: string; className: string; barColor: string }> = {
  active: {
    label: 'Active',
    className: 'bg-[hsl(var(--info))]/10 text-[hsl(var(--info))]',
    barColor: 'hsl(var(--info))',
  },
  completed: {
    label: 'Completed',
    className: 'bg-[hsl(var(--success))]/10 text-[hsl(var(--success))]',
    barColor: 'hsl(var(--success))',
  },
  paused: {
    label: 'Paused',
    className: 'bg-[hsl(var(--warning))]/10 text-[hsl(var(--warning))]',
    barColor: 'hsl(var(--warning))',
  },
  draft: {
    label: 'Draft',
    className: 'bg-secondary/50 text-muted-foreground',
    barColor: 'hsl(var(--muted-foreground))',
  },
}

const PRIORITY_DOT: Record<Project['priority'], string> = {
  urgent: 'bg-red-500',
  high: 'bg-orange-500',
  medium: 'bg-yellow-500',
  low: 'bg-green-500',
}

// ─── Component ──────────────────────────────────────────

interface ProjectCardProps {
  project: Project
  index: number
}

export function ProjectCard({ project, index }: ProjectCardProps) {
  const prefersReducedMotion = useReducedMotion()
  const statusConf = STATUS_CONFIG[project.status]
  const timeAgo = formatDistanceToNow(new Date(project.updated_at), { addSuffix: true })

  return (
    <motion.div
      initial={prefersReducedMotion ? false : { opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: index * 0.05 }}
      className="glass-card p-4 space-y-3 cursor-pointer hover:bg-secondary/20 transition-colors"
    >
      {/* Header: title + status */}
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <span className={cn('w-2 h-2 rounded-full shrink-0', PRIORITY_DOT[project.priority])} />
          <h3 className="font-semibold text-sm truncate">{project.name}</h3>
        </div>
        <Badge className={cn('text-[10px] shrink-0', statusConf.className)}>
          {statusConf.label}
        </Badge>
      </div>

      {/* Description */}
      <p className="text-xs text-muted-foreground line-clamp-2">{project.description}</p>

      {/* Progress bar */}
      <div>
        <div className="flex items-center justify-between text-[10px] text-muted-foreground mb-1">
          <span>Progress</span>
          <span>{project.progress}%</span>
        </div>
        <div className="h-1.5 bg-secondary/50 rounded-full overflow-hidden">
          <div
            className="h-full rounded-full transition-all duration-500"
            style={{
              width: `${project.progress}%`,
              backgroundColor: statusConf.barColor,
            }}
          />
        </div>
      </div>

      {/* Tags */}
      {project.tags.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {project.tags.slice(0, 3).map((tag) => (
            <span
              key={tag}
              className="text-[10px] px-1.5 py-0.5 rounded bg-secondary/50 text-muted-foreground"
            >
              {tag}
            </span>
          ))}
          {project.tags.length > 3 && (
            <span className="text-[10px] text-muted-foreground">+{project.tags.length - 3}</span>
          )}
        </div>
      )}

      {/* Footer: agent + task count + updated */}
      <div className="flex items-center justify-between pt-1">
        <div className="flex items-center gap-1.5 min-w-0">
          {project.lead_agent ? (
            <>
              {project.lead_agent.agent_icon ? (
                <PremiumIcon name={project.lead_agent.agent_icon} size={14} className="shrink-0" />
              ) : (
                <Bot className="w-3.5 h-3.5 text-primary shrink-0" />
              )}
              <span className="text-[10px] text-muted-foreground truncate">
                {project.lead_agent.agent_name}
              </span>
            </>
          ) : (
            <span className="text-[10px] text-muted-foreground/50">No lead agent</span>
          )}
        </div>

        <div className="flex items-center gap-3 shrink-0">
          <span className="flex items-center gap-1 text-[10px] text-muted-foreground">
            <CheckCircle2 className="w-3 h-3" />
            {project.tasks_completed}/{project.tasks_total}
          </span>
          <span className="text-[10px] text-muted-foreground">{timeAgo}</span>
        </div>
      </div>
    </motion.div>
  )
}
