'use client'

/**
 * ActivityTab — merged Feed + History stream for the Command Centre.
 *
 * One event stream from `useActivityFeed`, two densities (Cards / Table).
 * Type filter pills carry live counts; the `Errors` pill is tinted with
 * the burnt-orange accent to draw the eye.
 *
 * Cards render an agent avatar, status pip, timestamp, source, duration
 * and token-cost estimate. Table is the same data in a tabular layout for
 * dense scanning.
 */

import { useMemo, useState } from 'react'
import {
  Rows,
  Table2,
  Calendar as CalendarIcon,
  Pause,
  SlidersHorizontal,
  Play,
  Download,
  Wallet,
} from 'lucide-react'
import { useActivityFeed, type ActivityFeedItem } from '@/hooks/use-activity-api'
import { toneFor, initialFor } from './agent-tones'
import { formatDistanceToNowStrict } from 'date-fns'

type Density = 'cards' | 'table'

const TYPE_FILTERS: { key: string; label: string }[] = [
  { key: 'all', label: 'All' },
  { key: 'routine', label: 'Routines' },
  { key: 'task', label: 'Tool calls' },
  { key: 'mission', label: 'Missions' },
  { key: 'chat', label: 'Chats' },
  { key: 'recipe', label: 'Playbooks' },
]

function statusKind(s: ActivityFeedItem['status']):
  | 'ok'
  | 'err'
  | 'warn'
  | 'info'
  | 'run' {
  switch (s) {
    case 'completed':
      return 'ok'
    case 'failed':
      return 'err'
    case 'cancelled':
      return 'warn'
    case 'paused':
      return 'info'
    case 'pending':
      return 'info'
    case 'running':
    default:
      return 'run'
  }
}

function statusPip(s: ActivityFeedItem['status']): string {
  switch (s) {
    case 'completed':
      return '✓'
    case 'failed':
      return '!'
    case 'cancelled':
      return '×'
    case 'paused':
      return '||'
    case 'running':
      return '↻'
    default:
      return '·'
  }
}

function fmtRel(ts: string | null): string {
  if (!ts) return '—'
  try {
    return formatDistanceToNowStrict(new Date(ts), { addSuffix: false })
  } catch {
    return '—'
  }
}

function fmtClock(ts: string | null): string {
  if (!ts) return '—'
  try {
    return new Date(ts).toLocaleTimeString('en-GB', {
      hour: '2-digit',
      minute: '2-digit',
    })
  } catch {
    return '—'
  }
}

function fmtDuration(seconds: number | null): string {
  if (seconds == null) return '—'
  if (seconds < 60) return `${Math.round(seconds)}s`
  if (seconds < 3600) {
    const m = Math.floor(seconds / 60)
    return `${m}m`
  }
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  return `${h}h ${m}m`
}

function fmtTokens(tokens: number | null | undefined): string {
  if (!tokens) return '—'
  if (tokens >= 1_000_000) return `${(tokens / 1_000_000).toFixed(1)}M tok`
  if (tokens >= 1_000) return `${(tokens / 1_000).toFixed(1)}k tok`
  return `${tokens} tok`
}

export function ActivityTab() {
  const [density, setDensity] = useState<Density>('cards')
  const [typeFilter, setTypeFilter] = useState<string>('all')

  const filters = typeFilter === 'all' ? {} : { types: [typeFilter] }
  const { data, isLoading } = useActivityFeed({ ...filters, limit: 50 })

  const items = data?.items ?? []
  const totalShown = items.length

  // Count by type for the filter pills — uses an unfiltered fetch's totals if
  // the API exposes them. Otherwise we compute against the visible page so
  // counts make sense even on heavy filters.
  const { data: allData } = useActivityFeed({ limit: 200 })
  const counts = useMemo(() => {
    const all = allData?.items ?? items
    const total = all.length
    const errs = all.filter((i) => i.status === 'failed').length
    const byType = TYPE_FILTERS.reduce<Record<string, number>>((acc, f) => {
      acc[f.key] = f.key === 'all' ? total : all.filter((i) => i.type === f.key).length
      return acc
    }, {})
    return { total, errs, byType }
  }, [allData, items])

  return (
    <>
      <div className="cc-toolbar">
        <div className="cc-seg" role="group" aria-label="Density">
          <button
            type="button"
            className={density === 'cards' ? 'on' : ''}
            onClick={() => setDensity('cards')}
          >
            <Rows style={{ width: 11, height: 11 }} /> Cards
          </button>
          <button
            type="button"
            className={density === 'table' ? 'on' : ''}
            onClick={() => setDensity('table')}
          >
            <Table2 style={{ width: 11, height: 11 }} /> Table
          </button>
        </div>

        <div style={{ display: 'inline-flex', gap: 4, flexWrap: 'wrap' }}>
          {TYPE_FILTERS.map((f) => {
            const c = counts.byType[f.key] ?? 0
            const on = typeFilter === f.key
            return (
              <button
                key={f.key}
                type="button"
                className={`cc-filter-pill${on ? ' on' : ''}`}
                onClick={() => setTypeFilter(f.key)}
              >
                {f.label}
                <span className="ct">{c}</span>
              </button>
            )
          })}
          <button
            type="button"
            className={`cc-filter-pill err`}
            onClick={() => setTypeFilter('all')}
            title="Show only errors (toggles back to All)"
            disabled
          >
            Errors
            <span className="ct">{counts.errs}</span>
          </button>
        </div>

        <span style={{ marginLeft: 'auto', display: 'inline-flex', gap: 6 }}>
          <button type="button" className="cc-btn" style={{ height: 28, fontSize: 11.5 }}>
            <CalendarIcon style={{ width: 11, height: 11 }} />
            Today
          </button>
          <button type="button" className="cc-btn" style={{ height: 28, fontSize: 11.5 }}>
            <Play style={{ width: 11, height: 11 }} />
            Replay
          </button>
          <button type="button" className="cc-btn" style={{ height: 28, fontSize: 11.5 }}>
            <Download style={{ width: 11, height: 11 }} />
            Export CSV
          </button>
          <button type="button" className="cc-btn primary" style={{ height: 28, fontSize: 11.5 }}>
            <Wallet style={{ width: 11, height: 11 }} />
            Open in books
          </button>
        </span>
      </div>

      <div className="cc-panel" style={{ flex: 1, minHeight: 0 }}>
        <div className="cc-panel-head">
          <span className="t">Live stream</span>
          <span className="meta">
            · {totalShown} shown of {data?.total ?? totalShown} · tick 60s
          </span>
          <span
            style={{
              marginLeft: 12,
              display: 'inline-flex',
              gap: 4,
              alignItems: 'center',
              fontFamily: 'var(--font-geist-mono, monospace)',
              fontSize: 10,
              color: 'hsl(82 50% 22%)',
            }}
          >
            <span
              style={{
                width: 6,
                height: 6,
                borderRadius: '50%',
                background: 'hsl(82 50% 22%)',
              }}
            />
            LIVE
          </span>
          <div className="right">
            <button
              type="button"
              className="cc-btn"
              style={{ height: 26, fontSize: 11.5 }}
            >
              <Pause style={{ width: 11, height: 11 }} />
              Pause
            </button>
            <button
              type="button"
              className="cc-btn"
              style={{ height: 26, fontSize: 11.5 }}
            >
              <SlidersHorizontal style={{ width: 11, height: 11 }} />
              Columns
            </button>
          </div>
        </div>

        <div style={{ flex: 1, minHeight: 0, overflow: 'auto' }}>
          {isLoading ? (
            <div className="cc-panel-empty">Loading activity…</div>
          ) : items.length === 0 ? (
            <div className="cc-panel-empty">
              No events in this window. Adjust the filter or extend the time range.
            </div>
          ) : density === 'cards' ? (
            <CardsView items={items} />
          ) : (
            <TableView items={items} />
          )}
        </div>
      </div>
    </>
  )
}

function CardsView({ items }: { items: ActivityFeedItem[] }) {
  return (
    <div className="cc-act-cards">
      {items.map((it) => {
        const agName = it.agent?.name || (it.agents?.[0]?.name ?? 'System')
        const tone = toneFor(agName)
        const init = initialFor(agName)
        const kind = statusKind(it.status)
        const pip = statusPip(it.status)
        return (
          <div className="cc-act-card" key={it.id}>
            <span className={`pip ${kind}`}>{pip}</span>
            <div style={{ minWidth: 0 }}>
              <div className="nm">{it.name}</div>
              {it.summary && <div className="det">{it.summary}</div>}
              {it.error_message && (
                <div
                  className="det"
                  style={{ color: 'hsl(var(--accent))', marginTop: 2 }}
                >
                  {it.error_message}
                </div>
              )}
            </div>
            <div className="ag">
              <span className="av" style={{ background: tone.bg }}>
                {init}
              </span>
              <span>{agName}</span>
            </div>
            <span className="ts">
              {fmtClock(it.started_at)} · {it.type}
            </span>
            <span className="dur">{fmtDuration(it.duration_seconds)}</span>
            <span className="cost">{fmtTokens(it.tokens_used ?? it.total_tokens)}</span>
          </div>
        )
      })}
    </div>
  )
}

function TableView({ items }: { items: ActivityFeedItem[] }) {
  return (
    <table className="cc-act-table">
      <thead>
        <tr>
          <th style={{ width: 84 }}>When</th>
          <th style={{ width: 90 }}>Source</th>
          <th>Event</th>
          <th style={{ width: 140 }}>Agent</th>
          <th style={{ width: 88 }}>Status</th>
          <th style={{ width: 84, textAlign: 'right' }}>Duration</th>
          <th style={{ width: 88, textAlign: 'right' }}>Tokens</th>
        </tr>
      </thead>
      <tbody>
        {items.map((it) => {
          const agName = it.agent?.name || (it.agents?.[0]?.name ?? 'System')
          const tone = toneFor(agName)
          const kind = statusKind(it.status)
          return (
            <tr key={it.id}>
              <td className="mono">
                <div>{fmtClock(it.started_at)}</div>
                <div style={{ fontSize: 10, color: 'hsl(var(--muted-foreground))' }}>
                  {fmtRel(it.started_at)} ago
                </div>
              </td>
              <td className="mono" style={{ color: 'hsl(var(--foreground) / 0.7)' }}>
                {it.type}
              </td>
              <td>
                <div className="nm">{it.name}</div>
                {it.summary && (
                  <div
                    style={{
                      fontFamily: 'var(--font-geist-mono, monospace)',
                      fontSize: 10.5,
                      color: 'hsl(var(--muted-foreground))',
                      marginTop: 2,
                    }}
                  >
                    {it.summary}
                  </div>
                )}
              </td>
              <td>
                <span
                  aria-hidden
                  style={{
                    display: 'inline-block',
                    width: 8,
                    height: 8,
                    borderRadius: 2,
                    background: tone.bg,
                    marginRight: 8,
                    verticalAlign: 'middle',
                  }}
                />
                <span className="ag">{agName}</span>
              </td>
              <td>
                <span className={`cc-status-pill ${kind}`}>
                  ● {kind.toUpperCase()}
                </span>
              </td>
              <td className="mono" style={{ textAlign: 'right' }}>
                {fmtDuration(it.duration_seconds)}
              </td>
              <td className="mono" style={{ textAlign: 'right' }}>
                {fmtTokens(it.tokens_used ?? it.total_tokens)}
              </td>
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}
