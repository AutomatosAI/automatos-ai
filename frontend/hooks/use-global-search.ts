'use client'

import { useState, useEffect, useRef, useCallback, useMemo, type Dispatch, type SetStateAction } from 'react'
import { apiClient, type WorkspaceSocialsState } from '@/lib/api-client'
import { feedItemHref } from '@/lib/ticket-links'
import { SOCIAL_STATUS_LABELS } from '@/components/deliverables/socials/socials-status'

const ACTIVITY_HREF = '/command-center?tab=activity'

export interface SearchResult {
  id: string
  label: string
  description?: string
  category: 'pages' | 'tasks' | 'agents' | 'memories' | 'socials'
  path: string
  icon?: string
  /** More text the result was found by (a post's brief): the dialog keeps it listed when the query matches it. */
  keywords?: string
}

// PRD-251 S2.1: the Socials page is listed only while the platform offers Socials.
const SOCIALS_PAGE_ID = 'nav-socials'

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
  { id: SOCIALS_PAGE_ID, label: 'Socials', category: 'pages', path: '/deliverables?tab=socials' },
  { id: 'nav-explorer', label: 'Explorer', category: 'pages', path: '/deliverables/explorer' },
]

/** The query waits this long after the last keystroke, and needs this many characters. */
const SEARCH_DEBOUNCE_MS = 300
const MIN_QUERY_CHARS = 2
/** Posts listed per search, as the other sources ask for `limit=5`. */
const SOCIAL_POSTS_SHOWN = 5

function filterPages(query: string, socialsAvailable: boolean): SearchResult[] {
  const pages = socialsAvailable ? NAVIGATION_PAGES : NAVIGATION_PAGES.filter((p) => p.id !== SOCIALS_PAGE_ID)
  if (!query) return pages
  const lower = query.toLowerCase()
  return pages.filter((p) => p.label.toLowerCase().includes(lower))
}

// Routes are the real backend prefixes: activity lives at /api/activity and
// agents at /api/agents (NOT /api/v1/*, which 404s). Errors propagate to the
// caller — the hook surfaces them rather than silently returning [].
async function searchTasks(query: string): Promise<SearchResult[]> {
  const data = await apiClient.request<any>(`/api/activity/feed?search=${encodeURIComponent(query)}&limit=5`)
  const items = Array.isArray(data) ? data : data?.items ?? data?.data ?? []
  // PRD-252 R1: a feed item's id is "task-<id>", not the ticket's id (that is
  // source_id), so `task_id=${t.id}` never opened a ticket. Each row opens the
  // thing itself, as it does on the Activity tab.
  return items.map((t: any) => ({
    id: `task-${t.id}`,
    label: t.name || t.title || 'Untitled Task',
    description: [t.status, t.agent_name].filter(Boolean).join(' · '),
    category: 'tasks' as const,
    path: feedItemHref(t) ?? ACTIVITY_HREF,
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

// PRD-251 S2.1: the posts whose title or brief holds the query (GET /api/socials/posts?q=),
// each opening in the Socials tab. Asked only while Socials is on: the route is 404 otherwise (D1).
async function searchSocialPosts(query: string): Promise<SearchResult[]> {
  const { posts } = await apiClient.listSocialPosts({ q: query })
  return posts.slice(0, SOCIAL_POSTS_SHOWN).map((post) => ({
    id: `social-post-${post.id}`,
    label: post.title,
    description: SOCIAL_STATUS_LABELS[post.status],
    category: 'socials' as const,
    path: `/deliverables?tab=socials&post=${encodeURIComponent(post.id)}`,
    keywords: post.brief ?? undefined,
  }))
}

interface ApiResults {
  tasks: SearchResult[]
  agents: SearchResult[]
  memories: SearchResult[]
  socials: SearchResult[]
}

const NO_RESULTS: ApiResults = { tasks: [], agents: [], memories: [], socials: [] }

const settledResults = (outcome: PromiseSettledResult<SearchResult[]>): SearchResult[] =>
  outcome.status === 'fulfilled' ? outcome.value : []

async function searchSources(query: string, withSocials: boolean): Promise<{ results: ApiResults; failed: number }> {
  // allSettled, not all: one source failing must not blank the others, and
  // a failure is surfaced (error state) rather than silently swallowed.
  const settled = await Promise.allSettled([
    searchTasks(query),
    searchAgents(query),
    searchMemories(query),
    withSocials ? searchSocialPosts(query) : Promise.resolve<SearchResult[]>([]),
  ])
  const [tasks, agents, memories, socials] = settled.map(settledResults)
  const failed = settled.filter((outcome) => outcome.status === 'rejected').length
  return { results: { tasks, agents, memories, socials }, failed }
}

/** The API sources, searched once the query has settled; a newer query discards an older answer. */
function useSourceResults(query: string, withSocials: boolean) {
  const [results, setResults] = useState<ApiResults>(NO_RESULTS)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const generationRef = useRef(0)

  useEffect(() => {
    const generation = ++generationRef.current
    if (query.length < MIN_QUERY_CHARS) {
      setResults(NO_RESULTS)
      setError(null)
      setLoading(false)
      return
    }
    setLoading(true)
    const timer = setTimeout(async () => {
      const { results: found, failed } = await searchSources(query, withSocials)
      if (generationRef.current !== generation) return
      setResults(found)
      setError(failed > 0 ? 'Some results could not be loaded. Try again.' : null)
      setLoading(false)
    }, SEARCH_DEBOUNCE_MS)
    return () => clearTimeout(timer)
  }, [query, withSocials])

  return { results, loading, error }
}

/** ⌘K / Ctrl+K toggles the dialog; the Studio header's search button opens it with an event. */
function useSearchShortcut(setOpen: Dispatch<SetStateAction<boolean>>) {
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
  }, [setOpen])
}

/**
 * The global search. `socials` is the workspace's Socials state (PRD-251 D1): the
 * Socials page is listed while the platform offers it, and posts are searched while
 * this workspace has it on too. Without it (outside a workspace), neither.
 */
export function useGlobalSearch(socials?: WorkspaceSocialsState | null) {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const socialsAvailable = socials?.available === true
  const socialsOn = socialsAvailable && socials?.enabled === true
  const pages = useMemo(() => filterPages(query, socialsAvailable), [query, socialsAvailable])
  const { results, loading, error } = useSourceResults(query, socialsOn)
  useSearchShortcut(setOpen)

  // Reset on close: the cleared query clears the results.
  const handleOpenChange = useCallback((next: boolean) => {
    setOpen(next)
    if (!next) setQuery('')
  }, [])

  return { open, query, setQuery, loading, error, pages, ...results, handleOpenChange }
}
