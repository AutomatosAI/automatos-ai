'use client'

/**
 * MissionsBody — the toolbar + view body for the Studio Missions queue.
 *
 * Used standalone on /missions (wrapped with editorial top) and inside
 * the Assignments hub flip-tab. Keeps view/filter/search state local.
 */

import { useMemo, useState } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { useUser } from '@clerk/nextjs'
import { LayoutGrid, LayoutList, Table2, Search } from 'lucide-react'

import { useMissions } from '@/hooks/use-missions-api'
import type { MissionResponse, RunState } from '@/types/missions'

import { StatusHead } from './status-head'
import { MissionCard } from './mission-card'
import { MissionRow } from './mission-row'

type View = 'grouped' | 'cards' | 'table'
type ChipKey = 'all' | 'mine' | 'auto' | 'failed'

const STATUS_GROUPS: { label: string; states: RunState[]; pip: string }[] = [
  { label: 'Needs your call', states: ['awaiting_approval', 'awaiting_human'], pip: 'hsl(38 78% 50%)' },
  { label: 'Running',         states: ['running', 'planning', 'verifying', 'paused'], pip: 'hsl(var(--foreground))' },
  { label: 'Drafts',          states: ['pending'], pip: 'hsl(var(--muted-foreground))' },
  { label: 'Failed',          states: ['failed'], pip: 'hsl(var(--accent))' },
  { label: 'Completed',       states: ['completed'], pip: 'hsl(82 50% 22%)' },
  { label: 'Cancelled',       states: ['cancelled'], pip: 'hsl(40 30% 70%)' },
]

const STATE_TONE: Record<RunState, 'ok' | 'err' | 'warn' | 'info' | 'run' | 'muted'> = {
  pending: 'muted',
  planning: 'info',
  awaiting_approval: 'warn',
  running: 'run',
  paused: 'info',
  verifying: 'info',
  awaiting_human: 'warn',
  completed: 'ok',
  failed: 'err',
  cancelled: 'muted',
}

const STATE_LABEL: Record<RunState, string> = {
  pending: 'PENDING',
  planning: 'PLANNING',
  awaiting_approval: 'NEEDS APPROVAL',
  running: 'RUNNING',
  paused: 'PAUSED',
  verifying: 'VERIFYING',
  awaiting_human: 'NEEDS YOU',
  completed: 'COMPLETED',
  failed: 'FAILED',
  cancelled: 'CANCELLED',
}

function shortId(id: string) {
  const t = id.replace(/-/g, '').toUpperCase()
  return t.length > 6 ? t.slice(0, 6) : t
}

function fmtTokens(n: number) {
  if (!n) return '0'
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`
  return String(n)
}

interface MissionsBodyProps {
  /** Limit the queries — smaller fetch when embedded in the hub. */
  limit?: number
  /** Hide the status groups that have zero items (default true). */
  hideEmptyGroups?: boolean
}

export function MissionsBody({ limit = 100, hideEmptyGroups = true }: MissionsBodyProps) {
  const router = useRouter()
  const searchParams = useSearchParams()
  const { user } = useUser()

  const stateParam = (searchParams?.get('state') as RunState | null) ?? null

  const [view, setView] = useState<View>('grouped')
  const [chip, setChip] = useState<ChipKey>(stateParam === 'failed' ? 'failed' : 'all')
  const [search, setSearch] = useState('')

  const { data: missionData, isLoading } = useMissions({ limit })

  const allMissions = missionData?.missions ?? []
  const userKey = user?.id || user?.primaryEmailAddress?.emailAddress || ''

  const counts = useMemo(() => {
    const failed = allMissions.filter((m) => m.state === 'failed').length
    const mine = userKey
      ? allMissions.filter((m) => m.created_by === userKey).length
      : 0
    const auto = allMissions.filter(
      (m) => (m.created_by || '').toLowerCase() === 'auto',
    ).length
    return { all: allMissions.length, mine, auto, failed }
  }, [allMissions, userKey])

  const filtered = useMemo(() => {
    let list = allMissions
    if (chip === 'mine' && userKey) list = list.filter((m) => m.created_by === userKey)
    else if (chip === 'auto') list = list.filter((m) => (m.created_by || '').toLowerCase() === 'auto')
    else if (chip === 'failed') list = list.filter((m) => m.state === 'failed')

    if (search.trim()) {
      const q = search.trim().toLowerCase()
      list = list.filter(
        (m) =>
          m.goal.toLowerCase().includes(q) ||
          m.id.toLowerCase().includes(q) ||
          (m.created_by || '').toLowerCase().includes(q),
      )
    }
    return list
  }, [allMissions, chip, search, userKey])

  const handleOpen = (id: string) => router.push(`/missions/${id}` as any)

  return (
    <>
      {/* Toolbar */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 10,
          flexWrap: 'wrap',
        }}
      >
        <div className="view-toggle" role="group" aria-label="View">
          <button
            type="button"
            className={view === 'grouped' ? 'on' : ''}
            onClick={() => setView('grouped')}
          >
            <LayoutList style={{ width: 11, height: 11 }} />
            Grouped
          </button>
          <button
            type="button"
            className={view === 'cards' ? 'on' : ''}
            onClick={() => setView('cards')}
          >
            <LayoutGrid style={{ width: 11, height: 11 }} />
            Cards
          </button>
          <button
            type="button"
            className={view === 'table' ? 'on' : ''}
            onClick={() => setView('table')}
          >
            <Table2 style={{ width: 11, height: 11 }} />
            Table
          </button>
        </div>

        <div style={{ display: 'inline-flex', gap: 6, flexWrap: 'wrap' }}>
          {(
            [
              { k: 'all', l: 'All', c: counts.all },
              { k: 'mine', l: 'Mine', c: counts.mine },
              { k: 'auto', l: 'Auto-driven', c: counts.auto },
              { k: 'failed', l: 'Failed only', c: counts.failed, tone: 'err' as const },
            ] as { k: ChipKey; l: string; c: number; tone?: 'err' }[]
          ).map((p) => {
            const on = chip === p.k
            return (
              <button
                key={p.k}
                type="button"
                className={`studio-pill ${on ? 'run' : p.tone ?? 'muted'}`}
                onClick={() => setChip(p.k)}
                style={{
                  cursor: 'pointer',
                  background: on ? 'var(--fg)' : undefined,
                  color: on ? 'hsl(var(--background))' : undefined,
                  padding: '4px 10px',
                  fontSize: 10.5,
                }}
              >
                {p.l}
                <span style={{ opacity: 0.7, fontSize: 10 }}>{p.c}</span>
              </button>
            )
          })}
        </div>

        <div style={{ marginLeft: 'auto', display: 'inline-flex', alignItems: 'center', gap: 8 }}>
          <label
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: 6,
              background: 'var(--paper)',
              border: '1px solid hsl(var(--border))',
              borderRadius: 6,
              padding: '0 10px',
              height: 28,
              minWidth: 220,
            }}
          >
            <Search style={{ width: 12, height: 12, color: 'var(--fg-3)' }} />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Find by name, ID, owner…"
              style={{
                border: 0,
                outline: 'none',
                background: 'transparent',
                fontSize: 12,
                color: 'var(--fg)',
                flex: 1,
              }}
            />
          </label>
        </div>
      </div>

      {/* Body */}
      <div style={{ flex: 1, minHeight: 0 }}>
        {isLoading ? (
          <div
            className="pg-panel"
            style={{
              padding: 24,
              textAlign: 'center',
              color: 'var(--fg-3)',
              fontFamily: 'var(--font-geist-mono, monospace)',
              fontSize: 12,
            }}
          >
            Loading missions…
          </div>
        ) : filtered.length === 0 ? (
          <div
            className="pg-panel"
            style={{
              padding: 24,
              textAlign: 'center',
              color: 'var(--fg-3)',
              fontFamily: 'var(--font-geist-mono, monospace)',
              fontSize: 12,
            }}
          >
            No missions match this filter.
          </div>
        ) : view === 'grouped' ? (
          <GroupedView
            missions={filtered}
            onOpen={handleOpen}
            hideEmptyGroups={hideEmptyGroups}
          />
        ) : view === 'cards' ? (
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(2, minmax(0, 1fr))',
              gap: 12,
            }}
          >
            {filtered.map((m) => (
              <MissionCard key={m.id} mission={m} onOpen={handleOpen} />
            ))}
          </div>
        ) : (
          <TableView missions={filtered} onOpen={handleOpen} />
        )}
      </div>
    </>
  )
}

function GroupedView({
  missions,
  onOpen,
  hideEmptyGroups,
}: {
  missions: MissionResponse[]
  onOpen: (id: string) => void
  hideEmptyGroups: boolean
}) {
  return (
    <>
      {STATUS_GROUPS.map((g) => {
        const items = missions.filter((m) => g.states.includes(m.state))
        if (items.length === 0 && hideEmptyGroups) return null
        return (
          <section key={g.label}>
            <StatusHead pip={g.pip} label={g.label} count={items.length} />
            {items.length === 0 ? (
              <div
                className="pg-panel"
                style={{
                  padding: 14,
                  fontFamily: 'var(--font-geist-mono, monospace)',
                  fontSize: 11.5,
                  color: 'var(--fg-3)',
                  marginBottom: 6,
                }}
              >
                Nothing here.
              </div>
            ) : (
              <div className="pg-panel" style={{ padding: 0, marginBottom: 6 }}>
                {items.map((m) => (
                  <MissionRow key={m.id} mission={m} onOpen={onOpen} />
                ))}
              </div>
            )}
          </section>
        )
      })}
    </>
  )
}

function TableView({
  missions,
  onOpen,
}: {
  missions: MissionResponse[]
  onOpen: (id: string) => void
}) {
  return (
    <div className="pg-panel" style={{ padding: 0 }}>
      <table className="act-table">
        <thead>
          <tr>
            <th style={{ width: 90 }}>ID</th>
            <th>Mission</th>
            <th style={{ width: 130 }}>Status</th>
            <th style={{ width: 140 }}>Created by</th>
            <th style={{ width: 110, textAlign: 'right' }}>Tokens</th>
            <th style={{ width: 110 }}>Updated</th>
          </tr>
        </thead>
        <tbody>
          {missions.map((m) => {
            const tone = STATE_TONE[m.state]
            return (
              <tr key={m.id} onClick={() => onOpen(m.id)}>
                <td className="mono">
                  <span style={{ color: 'hsl(var(--accent))', fontWeight: 700 }}>
                    {shortId(m.id)}
                  </span>
                </td>
                <td>
                  <div
                    className="nm"
                    style={{
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                      maxWidth: 480,
                    }}
                  >
                    {m.goal}
                  </div>
                </td>
                <td>
                  <span className={`studio-pill ${tone}`}>
                    ● {STATE_LABEL[m.state]}
                  </span>
                </td>
                <td>
                  <span className="ag">{m.created_by || '—'}</span>
                </td>
                <td className="mono" style={{ textAlign: 'right' }}>
                  {fmtTokens(m.tokens_used)}
                </td>
                <td className="mono" style={{ color: 'var(--fg-3)' }}>
                  {m.updated_at
                    ? new Date(m.updated_at).toLocaleDateString('en-GB', {
                        day: '2-digit',
                        month: 'short',
                      })
                    : '—'}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
