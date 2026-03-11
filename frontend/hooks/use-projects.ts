'use client'

import { useMemo, useState } from 'react'

// ─── Types ──────────────────────────────────────────────

export interface Project {
  id: string
  name: string
  description: string
  status: 'active' | 'completed' | 'paused' | 'draft'
  priority: 'urgent' | 'high' | 'medium' | 'low'
  progress: number
  tasks_total: number
  tasks_completed: number
  lead_agent?: { agent_id: number; agent_name: string; agent_icon?: string }
  tags: string[]
  created_at: string
  updated_at: string
}

export type ProjectStatus = Project['status'] | 'all'

// ─── Mock Data ──────────────────────────────────────────

const MOCK_PROJECTS: Project[] = [
  {
    id: 'proj_1',
    name: 'Q1 Board Deck Preparation',
    description:
      'Compile financials, market analysis, and product roadmap into an investor-ready board presentation for the Q1 review.',
    status: 'active',
    priority: 'urgent',
    progress: 62,
    tasks_total: 14,
    tasks_completed: 9,
    lead_agent: { agent_id: 12, agent_name: 'Strategist', agent_icon: 'Brain' },
    tags: ['finance', 'presentation', 'q1'],
    created_at: '2026-03-01T10:00:00Z',
    updated_at: '2026-03-11T08:45:00Z',
  },
  {
    id: 'proj_2',
    name: 'Market Research Report',
    description:
      'Deep-dive competitive analysis across 5 verticals with TAM/SAM/SOM sizing and trend forecasting.',
    status: 'active',
    priority: 'high',
    progress: 35,
    tasks_total: 10,
    tasks_completed: 4,
    lead_agent: { agent_id: 8, agent_name: 'Researcher', agent_icon: 'Search' },
    tags: ['research', 'competitive-intel'],
    created_at: '2026-03-04T14:30:00Z',
    updated_at: '2026-03-10T22:10:00Z',
  },
  {
    id: 'proj_3',
    name: 'Customer Onboarding Automation',
    description:
      'Design and implement automated email sequences, tutorial generation, and progress tracking for new customers.',
    status: 'paused',
    priority: 'medium',
    progress: 48,
    tasks_total: 18,
    tasks_completed: 9,
    lead_agent: { agent_id: 15, agent_name: 'Automator', agent_icon: 'Zap' },
    tags: ['automation', 'onboarding', 'email'],
    created_at: '2026-02-20T09:00:00Z',
    updated_at: '2026-03-07T16:30:00Z',
  },
  {
    id: 'proj_4',
    name: 'Weekly Newsletter Pipeline',
    description:
      'Curate industry news, draft copy, generate visuals, and schedule distribution every Friday at 9 AM.',
    status: 'completed',
    priority: 'low',
    progress: 100,
    tasks_total: 8,
    tasks_completed: 8,
    lead_agent: { agent_id: 22, agent_name: 'Writer', agent_icon: 'PenTool' },
    tags: ['content', 'newsletter', 'recurring'],
    created_at: '2026-02-10T11:00:00Z',
    updated_at: '2026-03-08T09:05:00Z',
  },
  {
    id: 'proj_5',
    name: 'Security Audit Playbook',
    description:
      'Build a reusable audit checklist covering OWASP Top 10, dependency scanning, and secrets rotation procedures.',
    status: 'draft',
    priority: 'high',
    progress: 0,
    tasks_total: 12,
    tasks_completed: 0,
    tags: ['security', 'compliance'],
    created_at: '2026-03-10T15:00:00Z',
    updated_at: '2026-03-10T15:00:00Z',
  },
]

// ─── Hook ───────────────────────────────────────────────

export function useProjects(statusFilter: ProjectStatus = 'all', search = '') {
  const [isLoading] = useState(false)

  const projects = useMemo(() => {
    let filtered = MOCK_PROJECTS

    if (statusFilter !== 'all') {
      filtered = filtered.filter((p) => p.status === statusFilter)
    }

    if (search.trim()) {
      const q = search.toLowerCase()
      filtered = filtered.filter(
        (p) =>
          p.name.toLowerCase().includes(q) ||
          p.description.toLowerCase().includes(q) ||
          p.tags.some((t) => t.toLowerCase().includes(q))
      )
    }

    return filtered
  }, [statusFilter, search])

  return { projects, isLoading }
}
