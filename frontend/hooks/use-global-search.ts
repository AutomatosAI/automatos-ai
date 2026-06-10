'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { apiClient } from '@/lib/api-client'

export interface SearchResult {
  id: string
  label: string
  description?: string
  category: 'pages' | 'tasks' | 'agents' | 'memories'
  path: string
  icon?: string
}

const NAVIGATION_PAGES: SearchResult[] = [
  { id: 'nav-command-centre', label: 'Command Center', category: 'pages', path: '/command-center' },
  { id: 'nav-assignments', label: 'Assignments', category: 'pages', path: '/assignments' },
  { id: 'nav-deliverables', label: 'Deliverables', category: 'pages', path: '/deliverables' },
  { id: 'nav-agents', label: 'Agents', category: 'pages', path: '/agents' },
  { id: 'nav-chat', label: 'Chat', category: 'pages', path: '/chat' },
  { id: 'nav-analytics', label: 'Analytics', category: 'pages', path: '/analytics' },
  { id: 'nav-marketplace', label: 'Marketplace', category: 'pages', path: '/marketplace' },
  { id: 'nav-settings', label: 'Settings', category: 'pages', path: '/settings' },
  { id: 'nav-board', label: 'Board', category: 'pages', path: '/command-center?tab=board' },
  { id: 'nav-calendar', label: 'Calendar', category: 'pages', path: '/command-center?tab=calendar' },
  { id: 'nav-feed', label: 'Feed', category: 'pages', path: '/command-center?tab=feed' },
  { id: 'nav-history', label: 'History', category: 'pages', path: '/command-center?tab=history' },
  { id: 'nav-playbooks', label: 'Playbooks', category: 'pages', path: '/assignments?tab=playbooks' },
  { id: 'nav-missions', label: 'Missions', category: 'pages', path: '/assignments?tab=missions' },
  { id: 'nav-blogs', label: 'Blogs', category: 'pages', path: '/deliverables?tab=blogs' },
  { id: 'nav-templates', label: 'Templates', category: 'pages', path: '/deliverables?tab=templates' },
  { id: 'nav-explorer', label: 'Explorer', category: 'pages', path: '/deliverables/explorer' },
]

function filterPages(query: string): SearchResult[] {
  if (!query) return NAVIGATION_PAGES
  const lower = query.toLowerCase()
  return NAVIGATION_PAGES.filter((p) => p.label.toLowerCase().includes(lower))
}

// Routes are the real backend prefixes: activity lives at /api/activity and
// agents at /api/agents (NOT /api/v1/*, which 404s). Errors propagate to the
// caller — the hook surfaces them rather than silently returning [].
async function searchTasks(query: string): Promise<SearchResult[]> {
  const data = await apiClient.request<any>(`/api/activity/feed?search=${encodeURIComponent(query)}&limit=5`)
  const items = Array.isArray(data) ? data : data?.items ?? data?.data ?? []
  return items.map((t: any) => ({
    id: `task-${t.id}`,
    label: t.name || t.title || 'Untitled Task',
    description: [t.status, t.agent_name].filter(Boolean).join(' · '),
    category: 'tasks' as const,
    path: `/command-center?tab=board&task_id=${t.id}`,
  }))
}

async function searchAgents(query: string): Promise<SearchResult[]> {
  const data = await apiClient.request<any>(`/api/agents?search=${encodeURIComponent(query)}&limit=5`)
  const items = Array.isArray(data) ? data : data?.agents ?? data?.data ?? []
  return items.map((a: any) => ({
    id: `agent-${a.id}`,
    label: a.name || 'Unnamed Agent',
    description: a.role || a.skill || undefined,
    category: 'agents' as const,
    path: `/agents?agent_id=${a.id}`,
  }))
}

async function searchMemories(query: string): Promise<SearchResult[]> {
  const data = await apiClient.request<any>(`/api/v1/memory/browse?query=${encodeURIComponent(query)}&limit=5`)
  const items = Array.isArray(data) ? data : data?.memories ?? data?.data ?? []
  return items.map((m: any) => {
    const content = m.memory || m.content || m.text || ''
    return {
      id: `memory-${m.id}`,
      label: content.length > 80 ? content.slice(0, 80) + '...' : content,
      description: m.created_at ? new Date(m.created_at).toLocaleDateString() : undefined,
      category: 'memories' as const,
      path: `/command-center?memory_id=${m.id}`,
    }
  })
}

export function useGlobalSearch() {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [pages, setPages] = useState<SearchResult[]>(NAVIGATION_PAGES)
  const [tasks, setTasks] = useState<SearchResult[]>([])
  const [agents, setAgents] = useState<SearchResult[]>([])
  const [memories, setMemories] = useState<SearchResult[]>([])
  const abortRef = useRef(0)

  // Keyboard shortcut + custom open event (Studio cmdK button uses the event)
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault()
        setOpen((prev) => !prev)
      }
    }
    const openHandler = () => setOpen(true)
    window.addEventListener('keydown', handler)
    window.addEventListener('automatos:global-search-open', openHandler)
    return () => {
      window.removeEventListener('keydown', handler)
      window.removeEventListener('automatos:global-search-open', openHandler)
    }
  }, [])

  // Reset on close
  const handleOpenChange = useCallback((next: boolean) => {
    setOpen(next)
    if (!next) {
      setQuery('')
      setTasks([])
      setAgents([])
      setMemories([])
      setError(null)
      setPages(NAVIGATION_PAGES)
    }
  }, [])

  // Debounced search
  useEffect(() => {
    setPages(filterPages(query))

    if (query.length < 2) {
      setTasks([])
      setAgents([])
      setMemories([])
      setError(null)
      setLoading(false)
      return
    }

    setLoading(true)
    const generation = ++abortRef.current

    const timer = setTimeout(async () => {
      // allSettled, not all: one source failing must not blank the others, and
      // a failure is surfaced (error state) rather than silently swallowed.
      const settled = await Promise.allSettled([
        searchTasks(query),
        searchAgents(query),
        searchMemories(query),
      ])
      if (abortRef.current !== generation) return
      const [t, a, m] = settled
      setTasks(t.status === 'fulfilled' ? t.value : [])
      setAgents(a.status === 'fulfilled' ? a.value : [])
      setMemories(m.status === 'fulfilled' ? m.value : [])
      const failed = settled.filter((r) => r.status === 'rejected').length
      setError(failed > 0 ? 'Some results could not be loaded. Try again.' : null)
      setLoading(false)
    }, 300)

    return () => clearTimeout(timer)
  }, [query])

  return { open, query, setQuery, loading, error, pages, tasks, agents, memories, handleOpenChange }
}
