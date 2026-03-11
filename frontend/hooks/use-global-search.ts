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
  { id: 'nav-activity', label: 'Activity', category: 'pages', path: '/activity' },
  { id: 'nav-agents', label: 'Agents', category: 'pages', path: '/agents' },
  { id: 'nav-chat', label: 'Chat', category: 'pages', path: '/chat' },
  { id: 'nav-analytics', label: 'Analytics', category: 'pages', path: '/analytics' },
  { id: 'nav-marketplace', label: 'Marketplace', category: 'pages', path: '/marketplace' },
  { id: 'nav-settings', label: 'Settings', category: 'pages', path: '/settings' },
  { id: 'nav-command-centre', label: 'Command Centre', category: 'pages', path: '/activity?tab=summary' },
  { id: 'nav-board', label: 'Board', category: 'pages', path: '/activity?tab=board' },
  { id: 'nav-calendar', label: 'Calendar', category: 'pages', path: '/activity?tab=calendar' },
  { id: 'nav-memory', label: 'Memory', category: 'pages', path: '/activity?tab=memory' },
  { id: 'nav-projects', label: 'Projects', category: 'pages', path: '/activity?tab=projects' },
]

function filterPages(query: string): SearchResult[] {
  if (!query) return NAVIGATION_PAGES
  const lower = query.toLowerCase()
  return NAVIGATION_PAGES.filter((p) => p.label.toLowerCase().includes(lower))
}

async function searchTasks(query: string): Promise<SearchResult[]> {
  try {
    const data = await apiClient.request<any>(`/api/v1/activity/feed?search=${encodeURIComponent(query)}&limit=5`)
    const items = Array.isArray(data) ? data : data?.items ?? data?.data ?? []
    return items.map((t: any) => ({
      id: `task-${t.id}`,
      label: t.name || t.title || 'Untitled Task',
      description: [t.status, t.agent_name].filter(Boolean).join(' · '),
      category: 'tasks' as const,
      path: `/activity?tab=board&task_id=${t.id}`,
    }))
  } catch {
    return []
  }
}

async function searchAgents(query: string): Promise<SearchResult[]> {
  try {
    const data = await apiClient.request<any>(`/api/v1/agents?search=${encodeURIComponent(query)}&limit=5`)
    const items = Array.isArray(data) ? data : data?.agents ?? data?.data ?? []
    return items.map((a: any) => ({
      id: `agent-${a.id}`,
      label: a.name || 'Unnamed Agent',
      description: a.role || a.skill || undefined,
      category: 'agents' as const,
      path: `/agents?agent_id=${a.id}`,
    }))
  } catch {
    return []
  }
}

async function searchMemories(query: string): Promise<SearchResult[]> {
  try {
    const data = await apiClient.request<any>(`/api/v1/memory/browse?query=${encodeURIComponent(query)}&limit=5`)
    const items = Array.isArray(data) ? data : data?.memories ?? data?.data ?? []
    return items.map((m: any) => {
      const content = m.memory || m.content || m.text || ''
      return {
        id: `memory-${m.id}`,
        label: content.length > 80 ? content.slice(0, 80) + '...' : content,
        description: m.created_at ? new Date(m.created_at).toLocaleDateString() : undefined,
        category: 'memories' as const,
        path: `/activity?tab=memory&memory_id=${m.id}`,
      }
    })
  } catch {
    return []
  }
}

export function useGlobalSearch() {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(false)
  const [pages, setPages] = useState<SearchResult[]>(NAVIGATION_PAGES)
  const [tasks, setTasks] = useState<SearchResult[]>([])
  const [agents, setAgents] = useState<SearchResult[]>([])
  const [memories, setMemories] = useState<SearchResult[]>([])
  const abortRef = useRef(0)

  // Keyboard shortcut
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault()
        setOpen((prev) => !prev)
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [])

  // Reset on close
  const handleOpenChange = useCallback((next: boolean) => {
    setOpen(next)
    if (!next) {
      setQuery('')
      setTasks([])
      setAgents([])
      setMemories([])
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
      setLoading(false)
      return
    }

    setLoading(true)
    const generation = ++abortRef.current

    const timer = setTimeout(async () => {
      const [t, a, m] = await Promise.all([
        searchTasks(query),
        searchAgents(query),
        searchMemories(query),
      ])
      if (abortRef.current !== generation) return
      setTasks(t)
      setAgents(a)
      setMemories(m)
      setLoading(false)
    }, 300)

    return () => clearTimeout(timer)
  }, [query])

  return { open, query, setQuery, loading, pages, tasks, agents, memories, handleOpenChange }
}
