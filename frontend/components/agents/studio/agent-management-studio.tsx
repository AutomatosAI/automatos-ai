'use client'

/**
 * PRD-244 W5a (D8) — Agent Management in the Studio style: the editorial
 * head, in-page tabs (the hub's pattern — the header sub-nav no longer lists
 * this page), an honest stats strip and the Studio toolbar, composing the
 * same tab bodies the Classic page mounts. Classic is untouched.
 */
import { useCallback, useEffect, useMemo, useState } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { LayoutGrid, List, Plus, RefreshCw } from 'lucide-react'
import { SearchInput } from '@/components/shared/search-input'
import { useViewMode } from '@/hooks/use-view-mode'
import { useWorkspace } from '@/components/workspace-provider'
import { useAgents, useAgentStats } from '@/hooks/use-agent-api'
import { AGENT_TAB_VALUES, type AgentTab } from '@/lib/agents/tabs'
import { summariseAgents } from '@/lib/agents/stats'
import { AgentRoster } from '../agent-roster'
import { AgentConfiguration } from '../agent-configuration'
import { WorkspaceSkillsTab } from '../skills/workspace-skills-tab'
import { CreateAgentModal } from '../create-agent-modal'
import { AgentDetailsModal } from '../agent-details-modal'
import { OrgChartTab } from '../org-chart-tab'

const TAB_LABELS: Record<AgentTab, string> = {
  roster: 'Roster',
  'org-chart': 'Org Chart',
  configuration: 'Configuration',
  skills: 'Skills',
}

const STATUS_FILTERS = [
  { key: 'all', label: 'All' },
  { key: 'active', label: 'Active' },
  { key: 'idle', label: 'Idle' },
] as const

type StatusFilter = (typeof STATUS_FILTERS)[number]['key']

function isAgentTab(v: string | null): v is AgentTab {
  return v !== null && (AGENT_TAB_VALUES as readonly string[]).includes(v)
}

export function AgentManagementStudio() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const { canEdit } = useWorkspace()
  const { data: agents = [], isLoading: agentsLoading, refetch: refetchAgents, error: agentsError } = useAgents()
  const { data: agentStats } = useAgentStats()
  const [viewMode, setViewMode] = useViewMode('agents')

  const requested = searchParams?.get('tab') ?? null
  // PRD-244 review: ?agent=<id>&panel=<tab> opens that agent's details (an
  // activity routine row lands on its Reports panel).
  const deepLinkAgent = searchParams?.get('agent') ?? null
  const deepLinkPanel = searchParams?.get('panel') ?? undefined
  const tab: AgentTab = isAgentTab(requested) ? requested : 'roster'
  const selectTab = useCallback(
    (next: AgentTab) => {
      const params = new URLSearchParams(searchParams?.toString() ?? '')
      params.set('tab', next)
      router.replace(`/agents?${params.toString()}` as any)
    },
    [router, searchParams],
  )

  const [searchTerm, setSearchTerm] = useState('')
  const [statusFilter, setStatusFilter] = useState<StatusFilter>('all')
  const [selectedAgentId, setSelectedAgentId] = useState<string | null>(null)
  const [viewDetailsAgentId, setViewDetailsAgentId] = useState<string | null>(null)
  const [showCreate, setShowCreate] = useState(false)

  useEffect(() => {
    if (deepLinkAgent) setViewDetailsAgentId(deepLinkAgent)
  }, [deepLinkAgent])

  const roster = agents as Array<{ id: number | string; status?: string | null }>
  const summary = useMemo(() => summariseAgents(roster, agentStats as any), [roster, agentStats])
  const counts = useMemo(
    () => ({
      all: roster.length,
      active: roster.filter((a) => a.status === 'active').length,
      idle: roster.filter((a) => a.status !== 'active').length,
    }),
    [roster],
  )

  // Configuration needs a subject: default to the first agent, like Classic.
  useEffect(() => {
    if (tab === 'configuration' && !selectedAgentId && roster.length > 0) {
      setSelectedAgentId(String(roster[0].id))
    }
  }, [tab, roster, selectedAgentId])

  const refresh = () => void refetchAgents()
  const viewDetails = (agentId: string | null) => {
    if (agentId) setViewDetailsAgentId(agentId)
  }

  const cells = [
    { label: 'TOTAL', value: String(summary.total), tone: 'info', delta: `${summary.total} agent${summary.total === 1 ? '' : 's'}` },
    { label: 'ACTIVE', value: String(summary.active), tone: summary.active > 0 ? 'ok' : '', delta: summary.total > 0 ? `${Math.round((summary.active / summary.total) * 100)}% online` : '0% online' },
    { label: 'ATTENTION', value: String(summary.attention), tone: summary.attention > 0 ? 'err' : '', delta: summary.failing > 0 ? `${summary.failing} failing` : 'All healthy' },
    { label: 'SUCCESS', value: summary.avgSuccess === null ? '—' : `${summary.avgSuccess.toFixed(1)}%`, tone: summary.avgSuccess === null ? '' : summary.avgSuccess >= 90 ? 'ok' : 'warn', delta: summary.avgSuccess === null ? 'no data yet' : summary.avgSuccess >= 90 ? 'healthy' : 'needs work' },
  ]

  return (
    <div className="cc-page">
      <div className="cc-headrow">
        <div className="cc-head">
          <p className="cc-eyebrow">
            Workforce · {TAB_LABELS[tab]} · {summary.total} agent{summary.total === 1 ? '' : 's'}
          </p>
          <h1 className="cc-h1">Agent Management</h1>
          <p className="cc-sub">
            Your agents, what they&apos;re good at, and what they&apos;re doing. Configure
            capabilities, swap models, install skills, retire the ones you no longer need.
          </p>
        </div>
        <div className="cc-actions">
          <button type="button" className="cc-btn" onClick={refresh} disabled={agentsLoading && !agentsError}>
            <RefreshCw style={{ width: 12, height: 12 }} className={agentsLoading ? 'animate-spin' : undefined} />
            Refresh
          </button>
          <button
            type="button"
            className="cc-btn"
            onClick={() => setShowCreate(true)}
            disabled={!canEdit}
            title={canEdit ? undefined : 'Viewers have read-only access'}
          >
            <Plus style={{ width: 12, height: 12 }} />
            Create agent
          </button>
        </div>
      </div>

      {!!agentsError && (
        <div className="cc-panel" role="alert" style={{ padding: 14 }}>
          <p className="cc-sub" style={{ margin: 0 }}>
            Agents failed to load (HTTP {(agentsError as { status?: number })?.status ?? 500}). Check the backend logs for <code>/api/agents</code>.
          </p>
          <button type="button" className="cc-btn" style={{ marginTop: 10 }} onClick={refresh}>
            Retry
          </button>
        </div>
      )}

      <nav className="cc-tabs" aria-label="Agent Management sections">
        {AGENT_TAB_VALUES.map((key) => (
          <button
            key={key}
            type="button"
            className={`cc-tab${tab === key ? ' active' : ''}`}
            aria-current={tab === key ? 'page' : undefined}
            onClick={() => selectTab(key)}
          >
            <span>{TAB_LABELS[key]}</span>
            {key === 'roster' && summary.total > 0 && <span className="cc-tab-ct">{summary.total}</span>}
          </button>
        ))}
      </nav>

      {tab === 'roster' && (
        <>
          <div className="cc-stats" aria-label="Agent statistics">
            {cells.map((c) => (
              <div key={c.label} className="cell">
                <div className="l">{c.label}</div>
                <div className={`v ${c.tone}`}>{c.value}</div>
                <span className="delta">{c.delta}</span>
              </div>
            ))}
          </div>

          <div className="cc-toolbar">
            <SearchInput
              value={searchTerm}
              onChange={setSearchTerm}
              placeholder="Search agents by name, type, or capabilities…"
              className="flex-1"
            />
            <div style={{ display: 'inline-flex', gap: 4, flexWrap: 'wrap' }} role="group" aria-label="Status">
              {STATUS_FILTERS.map((f) => (
                <button
                  key={f.key}
                  type="button"
                  className={`cc-filter-pill${statusFilter === f.key ? ' on' : ''}`}
                  onClick={() => setStatusFilter(f.key)}
                >
                  {f.label}
                  <span className="ct">{counts[f.key]}</span>
                </button>
              ))}
            </div>
            <div className="cc-seg" role="group" aria-label="View">
              <button type="button" className={viewMode === 'grid' ? 'on' : ''} onClick={() => setViewMode('grid')} aria-label="Grid view">
                <LayoutGrid style={{ width: 11, height: 11 }} /> Grid
              </button>
              <button type="button" className={viewMode === 'list' ? 'on' : ''} onClick={() => setViewMode('list')} aria-label="List view">
                <List style={{ width: 11, height: 11 }} /> List
              </button>
            </div>
          </div>

          <AgentRoster
            agents={agents as any[]}
            loading={agentsLoading && !agentsError}
            searchTerm={searchTerm}
            statusFilter={statusFilter}
            onAgentSelect={setSelectedAgentId}
            onViewDetails={viewDetails}
            selectedAgentId={selectedAgentId}
            onRefresh={refresh}
            setSearchTerm={setSearchTerm}
            viewMode={viewMode}
          />
        </>
      )}

      {tab === 'org-chart' && <OrgChartTab />}
      {tab === 'configuration' && (
        <AgentConfiguration agents={agents as any[]} selectedAgentId={selectedAgentId} onAgentSelect={setSelectedAgentId} />
      )}
      {tab === 'skills' && <WorkspaceSkillsTab viewMode={viewMode} />}

      {viewDetailsAgentId && (
        <AgentDetailsModal agentId={Number(viewDetailsAgentId)} open onClose={() => setViewDetailsAgentId(null)} initialTab={deepLinkPanel} />
      )}
      <CreateAgentModal
        open={showCreate}
        onClose={() => setShowCreate(false)}
        onSuccess={() => {
          setShowCreate(false)
          refresh()
        }}
      />
    </div>
  )
}
