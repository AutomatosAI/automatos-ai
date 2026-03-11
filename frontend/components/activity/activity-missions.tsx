'use client'

import { useState } from 'react'
import { Rocket, Search, Plus } from 'lucide-react'
import { motion, useReducedMotion } from 'framer-motion'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { cn } from '@/lib/utils'
import { useProjects, type ProjectStatus } from '@/hooks/use-projects'
import { ProjectCard } from './projects'

// ─── Filter Tabs ────────────────────────────────────────

const STATUS_FILTERS: { value: ProjectStatus; label: string }[] = [
  { value: 'all', label: 'All' },
  { value: 'active', label: 'Active' },
  { value: 'completed', label: 'Completed' },
  { value: 'paused', label: 'Paused' },
]

// ─── Component ──────────────────────────────────────────

export function ActivityMissions() {
  const prefersReducedMotion = useReducedMotion()
  const [statusFilter, setStatusFilter] = useState<ProjectStatus>('all')
  const [search, setSearch] = useState('')

  const { projects } = useProjects(statusFilter, search)

  return (
    <div className="space-y-4">
      {/* Filter bar */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center gap-3">
        <div className="flex items-center gap-1 p-1 rounded-lg bg-secondary/30">
          {STATUS_FILTERS.map((f) => (
            <button
              key={f.value}
              onClick={() => setStatusFilter(f.value)}
              className={cn(
                'px-3 py-1.5 text-xs font-medium rounded-md transition-colors',
                statusFilter === f.value
                  ? 'bg-primary text-primary-foreground'
                  : 'text-muted-foreground hover:text-foreground hover:bg-secondary/50'
              )}
            >
              {f.label}
            </button>
          ))}
        </div>

        <div className="relative flex-1 w-full sm:max-w-xs">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-muted-foreground" />
          <Input
            placeholder="Search projects..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="pl-8 h-8 text-sm bg-secondary/20"
          />
        </div>

        <Button variant="outline" size="sm" className="ml-auto shrink-0" disabled>
          <Plus className="w-3.5 h-3.5 mr-1.5" />
          New Project
        </Button>
      </div>

      {/* Project grid or empty state */}
      {projects.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {projects.map((project, i) => (
            <ProjectCard key={project.id} project={project} index={i} />
          ))}
        </div>
      ) : (
        <motion.div
          initial={prefersReducedMotion ? false : { opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
          className="glass-card p-8 md:p-12 max-w-lg mx-auto text-center space-y-4"
        >
          <Rocket className="w-10 h-10 mx-auto text-muted-foreground/30" />

          <div>
            <h3 className="text-lg font-semibold">Create your first project</h3>
            <p className="text-sm text-muted-foreground mt-1 max-w-sm mx-auto">
              Projects are multi-agent missions that run for hours or days. Give a complex
              brief and your AI workforce figures out the steps, assigns agents, and delivers
              results.
            </p>
          </div>

          <Button variant="outline" className="border-primary/30 hover:border-primary/50" disabled>
            <Rocket className="w-4 h-4 mr-2" />
            New Project
          </Button>
        </motion.div>
      )}
    </div>
  )
}
