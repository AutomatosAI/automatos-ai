'use client'

/**
 * ActivityTab — merged Feed + History stream for the Command Centre.
 *
 * One event stream from `useActivityFeed`. Two real interactions:
 *  - Density toggle (Cards / Table)
 *  - Filter pills toggle the `types` filter passed to the API; the Errors
 *    pill toggles a status=failed filter on top.
 *  - Clicking a row routes to the live execution viewer for chats /
 *    routines / recipes (the existing /activity/execution viewer the
 *    classic feed uses).
 *
 * No speculative Pause / Replay / Open-in-books / Columns buttons — those
 * weren't wired anywhere.
 */

import { useMemo, useState } from 'react'
import { useRouter } from 'next/navigation'
import { Rows, Table2 } from 'lucide-react'
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
function fmtRel(ts: string | null): string {
  if (!ts) return '—'
  try {
    return formatDistanceToNowStrict(new Date(ts), { addSuffix: false })
  } catch {
    return '—'
  }
}
function fmtDuration(seconds: number | null): string {
  if (seconds == null) return '—'
  if (seconds < 60) return `${Math.round(seconds)}s`
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m`
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

function rowHref(item: ActivityFeedItem): string | null {
  // Mirror the classic ActivityFeed's deep-link behaviour. Only playbooks
  // (recipes) belong in the ExecutionKitchen viewer. Routines are
  // heartbeats — they should go to the agent that owns them, not the
  // playbook viewer.
  switch (item.type) {
    case 'mission':
      return item.source_id ? `/missions/${item.source_id}` : null
    case 'chat':
      // Use query param so client-side auth loads the thread; /chat/[id]
      // server route 404s without a server-side auth context.
      return item.source_id
        ? `/chat?chatId=${encodeURIComponent(item.source_id)}`
        : null
    case 'recipe':
      if (!item.source_id) return null
      // Recipe id pattern: feed id is "recipe-<execId>", source_id is the recipe id
      return `/activity/execution?id=${encodeURIComponent(
        item.id.replace(/^recipe-/, ''),
      )}&recipeId=${encodeURIComponent(item.source_id)}`
    case 'task':
      return item.source_id
        ? `/command-center?tab=board&task=${encodeURIComponent(item.source_id)}`
        : null
    case 'routine':
      // Heartbeat — open the owning agent so the user can pause/tune it
      return item.agent?.id ? `/agents?agent=${item.agent.id}` : null
    default:
      return null
  }
}

export function ActivityTab() {
  const router = useRouter()
  const [density, setDensity] = useState<Density>('cards')
  const [typeFilter, setTypeFilter] = useState<string>('all')
  const [errorsOnly, setErrorsOnly] = useState(false)

  const filters = {
    ...(typeFilter !== 'all' ? { types: [typeFilter] } : {}),
    ...(errorsOnly ? { status: 'failed' } : {}),
    limit: 100,
  }
  const { data, isLoading } = useActivityFeed(filters)
  const items = data?.items ?? []
  const totalShown = items.length

  // Unfiltered fetch just to power the filter-pill counts.
  // Backend caps limit at 100; sending 200 returns 422 and refetches
  // on every window-focus event, which made the page flicker repeatedly.
  const { data: allData } = useActivityFeed({ limit: 100 })
  const counts = useMemo(() => {
    const all = allData?.items ?? items
    const errs = all.filter((i) => i.status === 'failed').length
    const byType = TYPE_FILTERS.reduce<Record<string, number>>((acc, f) => {
      acc[f.key] = f.key === 'all' ? all.length : all.filter((i) => i.type === f.key).length
      return acc
    }, {})
    return { errs, byType }
  }, [allData, items])

  const handleRowClick = (item: ActivityFeedItem) => {
    const href = rowHref(item)
    if (href) router.push(href as any)
  }

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
            className={`cc-filter-pill err${errorsOnly ? ' on' : ''}`}
            onClick={() => setErrorsOnly((v) => !v)}
            title="Toggle errors-only filter"
          >
            Errors
            <span className="ct">{counts.errs}</span>
          </button>
        </div>
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
        </div>

        <div style={{ flex: 1, minHeight: 0, overflow: 'auto' }}>
          {isLoading ? (
            <div className="cc-panel-empty">Loading activity…</div>
          ) : items.length === 0 ? (
            <div className="cc-panel-empty">
              No events in this window. Adjust the filter or wait for more
              activity.
            </div>
          ) : density === 'cards' ? (
            <CardsView items={items} onRowClick={handleRowClick} />
          ) : (
            <TableView items={items} onRowClick={handleRowClick} />
          )}
        </div>
      </div>
    </>
  )
}

function CardsView({
  items,
  onRowClick,
}: {
  items: ActivityFeedItem[]
  onRowClick: (it: ActivityFeedItem) => void
}) {
  return (
    <div className="cc-act-cards">
      {items.map((it) => {
        const agName = it.agent?.name || it.agents?.[0]?.name || 'System'
        const tone = toneFor(agName)
        const init = initialFor(agName)
        const kind = statusKind(it.status)
        const pip = statusPip(it.status)
        return (
          <button
            type="button"
            key={it.id}
            className="cc-act-card"
            onClick={() => onRowClick(it)}
          >
            <span className={`pip ${kind}`}>{pip}</span>
            <div style={{ minWidth: 0, textAlign: 'left' }}>
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
            <span className="cost">
              {fmtTokens(it.tokens_used ?? it.total_tokens)}
            </span>
          </button>
        )
      })}
    </div>
  )
}

function TableView({
  items,
  onRowClick,
}: {
  items: ActivityFeedItem[]
  onRowClick: (it: ActivityFeedItem) => void
}) {
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
          const agName = it.agent?.name || it.agents?.[0]?.name || 'System'
          const tone = toneFor(agName)
          const kind = statusKind(it.status)
          return (
            <tr
              key={it.id}
              onClick={() => onRowClick(it)}
              style={{ cursor: 'pointer' }}
            >
              <td className="mono">
                <div>{fmtClock(it.started_at)}</div>
                <div
                  style={{
                    fontSize: 10,
                    color: 'hsl(var(--muted-foreground))',
                  }}
                >
                  {fmtRel(it.started_at)} ago
                </div>
              </td>
              <td
                className="mono"
                style={{ color: 'hsl(var(--foreground) / 0.7)' }}
              >
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
