'use client'

/**
 * PlaybooksBody — the toolbar + featured strip + library for the
 * Studio Playbooks library. Used standalone on /playbooks and inside
 * the Assignments hub flip-tab.
 */

import { useMemo, useState } from 'react'
import { useRouter } from 'next/navigation'
import { useUser } from '@/lib/auth-hooks'
import {
  LayoutGrid,
  List as ListIcon,
  Search,
  ArrowDownNarrowWide,
  BookMarked,
  Clock,
  Zap as TriggerZap,
  Hand,
} from 'lucide-react'

import { useWorkflowPlaybooks, useExecutePlaybook } from '@/hooks/use-playbook-api'
import { toast } from 'sonner'

import { StatusHead } from './status-head'
import { PlaybookCard, type PlaybookLike } from './playbook-card'

type View = 'grid' | 'list'
type Scope = 'all' | 'mine' | 'workspace' | 'imported'
type Sort = 'most_run' | 'recent' | 'success'

function trigger(p: PlaybookLike): { label: string; kind: 'cron' | 'webhook' | 'manual' } {
  const t = (p.trigger_type || '').toLowerCase()
  if (t === 'cron' || p.schedule_config?.cron) {
    return { label: p.schedule_config?.cron || 'scheduled', kind: 'cron' }
  }
  if (t === 'webhook') return { label: 'webhook', kind: 'webhook' }
  return { label: 'manual', kind: 'manual' }
}

interface PlaybooksBodyProps {
  limit?: number
  /** Hide the featured strip (e.g. when embedded under a hub that already shows them) */
  hideFeatured?: boolean
}

export function PlaybooksBody({ limit = 100, hideFeatured = false }: PlaybooksBodyProps) {
  const router = useRouter()
  const { user } = useUser()

  const [view, setView] = useState<View>('grid')
  const [scope, setScope] = useState<Scope>('all')
  const [sort, setSort] = useState<Sort>('most_run')
  const [search, setSearch] = useState('')

  const { data, isLoading } = useWorkflowPlaybooks({ limit })
  const all = (data as { items?: PlaybookLike[] } | undefined)?.items ?? []

  // Identify the current user against the playbook `created_by` field
  // (set server-side to email or "system"). Same approach as missions-body.
  const userKey = user?.primaryEmailAddress?.emailAddress || user?.id || ''
  const userIdNum = user?.id ? Number(user.id) : NaN

  type RawPlaybook = PlaybookLike & {
    created_by?: string
    created_by_user_id?: number | null
    workspace_id?: string | null
    cloned_from_id?: number | null
    owner_type?: string
    is_marketplace_item?: boolean
  }

  const isMine = (p: RawPlaybook): boolean => {
    if (!userKey && Number.isNaN(userIdNum)) return false
    if (p.created_by && userKey && p.created_by === userKey) return true
    if (p.created_by_user_id != null && !Number.isNaN(userIdNum)
        && p.created_by_user_id === userIdNum) return true
    return false
  }

  const isImported = (p: RawPlaybook): boolean =>
    (p.cloned_from_id != null) ||
    p.owner_type === 'marketplace' ||
    p.is_marketplace_item === true

  const isWorkspace = (p: RawPlaybook): boolean =>
    p.workspace_id != null && !isMine(p) && !isImported(p)

  const counts = useMemo(() => {
    const list = all as RawPlaybook[]
    return {
      all: list.length,
      mine: list.filter(isMine).length,
      workspace: list.filter(isWorkspace).length,
      imported: list.filter(isImported).length,
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [all, userKey, userIdNum])

  const filtered = useMemo(() => {
    let list = all as RawPlaybook[]
    if (scope === 'mine') list = list.filter(isMine)
    else if (scope === 'workspace') list = list.filter(isWorkspace)
    else if (scope === 'imported') list = list.filter(isImported)
    if (search.trim()) {
      const q = search.trim().toLowerCase()
      list = list.filter(
        (p) =>
          (p.name || '').toLowerCase().includes(q) ||
          (p.description || '').toLowerCase().includes(q) ||
          (p.category || '').toLowerCase().includes(q),
      )
    }
    const sorted = [...list].sort((a, b) => {
      if (sort === 'recent') {
        const ta = (a as { updated_at?: string }).updated_at || ''
        const tb = (b as { updated_at?: string }).updated_at || ''
        return tb.localeCompare(ta)
      }
      if (sort === 'success') {
        return (b.success_rate ?? 0) - (a.success_rate ?? 0)
      }
      return (b.use_count ?? 0) - (a.use_count ?? 0)
    })
    return sorted
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [all, scope, sort, search, userKey, userIdNum])

  const featured = useMemo(() => all.filter((p) => p.is_featured).slice(0, 5), [all])
  const totalRuns = all.reduce((acc, p) => acc + (p.use_count ?? 0), 0)

  const handleRun = (p: PlaybookLike) => {
    const id = p.template_id || (p.id != null ? String(p.id) : '')
    if (id) router.push(`/playbooks?id=${encodeURIComponent(id)}` as any)
  }

  const executePlaybook = useExecutePlaybook()

  const handleFeaturedRun = async (p: PlaybookLike) => {
    const id = p.template_id || (p.id != null ? String(p.id) : '')
    if (!id) return
    try {
      const result: any = await executePlaybook.mutateAsync({ playbookId: id })
      toast('Playbook started', { description: `"${p.name}" is now running.` })
      if (result?.recipe_execution_id) {
        router.push(
          `/activity/execution?id=${encodeURIComponent(result.recipe_execution_id)}&recipeId=${encodeURIComponent(id)}` as any,
        )
      }
    } catch (err: any) {
      toast.error('Could not start playbook', { description: err?.message || 'Execution failed' })
    }
  }

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
        <div className="scope-tabs" role="group" aria-label="Scope">
          {(
            [
              { k: 'all', l: 'All', c: counts.all },
              { k: 'mine', l: 'Mine', c: counts.mine },
              { k: 'workspace', l: 'Workspace', c: counts.workspace },
              { k: 'imported', l: 'Imported', c: counts.imported },
            ] as { k: Scope; l: string; c: number }[]
          ).map((s) => (
            <button
              key={s.k}
              type="button"
              className={scope === s.k ? 'on' : ''}
              onClick={() => setScope(s.k)}
            >
              {s.l}
              <span className="ct">{s.c}</span>
            </button>
          ))}
        </div>

        <div className="view-toggle" role="group" aria-label="View">
          <button
            type="button"
            className={view === 'grid' ? 'on' : ''}
            onClick={() => setView('grid')}
          >
            <LayoutGrid style={{ width: 11, height: 11 }} />
            Grid
          </button>
          <button
            type="button"
            className={view === 'list' ? 'on' : ''}
            onClick={() => setView('list')}
          >
            <ListIcon style={{ width: 11, height: 11 }} />
            List
          </button>
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
              minWidth: 200,
            }}
          >
            <Search style={{ width: 12, height: 12, color: 'var(--fg-3)' }} />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search playbooks…"
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
          <button
            type="button"
            className="l-btn"
            onClick={() =>
              setSort((s) =>
                s === 'most_run' ? 'recent' : s === 'recent' ? 'success' : 'most_run',
              )
            }
            title="Toggle sort"
          >
            <ArrowDownNarrowWide style={{ width: 11, height: 11 }} />
            Sort:{' '}
            {sort === 'most_run'
              ? 'most run'
              : sort === 'recent'
              ? 'recent'
              : 'success'}
          </button>
        </div>
      </div>

      {/* Featured strip */}
      {!hideFeatured && featured.length > 0 && (
        <section>
          <StatusHead
            pip="hsl(82 50% 22%)"
            label="★ Featured · proven in the wild"
            count={`${featured.length} picks`}
          />
          <div className="mkt-strip">
            {featured.map((p) => (
              <button
                key={String(p.template_id || p.id)}
                type="button"
                className="mkt-card"
                onClick={() => handleFeaturedRun(p)}
                disabled={executePlaybook.isPending}
              >
                <div className="top">
                  <span className="ic">
                    <BookMarked style={{ width: 14, height: 14, strokeWidth: 1.6 }} />
                  </span>
                  <span className="nm">{p.name || 'Untitled'}</span>
                </div>
                <span className="tag">★ Featured</span>
                {p.description && <div className="desc">{p.description}</div>}
                <div className="foot">
                  <span>{p.category || 'playbook'}</span>
                  <span className="dl">{p.use_count ?? 0} runs</span>
                </div>
              </button>
            ))}
          </div>
        </section>
      )}

      {/* Library */}
      <section>
        <StatusHead
          pip="hsl(var(--foreground))"
          label="Your library"
          count={`${all.length} playbooks · ${totalRuns.toLocaleString()} total runs`}
        />
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
            Loading playbooks…
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
            No playbooks match this filter.
          </div>
        ) : view === 'grid' ? (
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(3, minmax(0, 1fr))',
              gap: 12,
              paddingBottom: 24,
            }}
          >
            {filtered.map((p) => (
              <PlaybookCard
                key={String(p.template_id || p.id)}
                playbook={p}
                onRun={handleRun}
              />
            ))}
          </div>
        ) : (
          <ListView items={filtered} onRun={handleRun} />
        )}
      </section>
    </>
  )
}

function ListView({
  items,
  onRun,
}: {
  items: PlaybookLike[]
  onRun: (p: PlaybookLike) => void
}) {
  return (
    <div className="pg-panel" style={{ padding: 0 }}>
      <table className="act-table">
        <thead>
          <tr>
            <th style={{ width: 38 }}></th>
            <th>Playbook</th>
            <th style={{ width: 160 }}>Trigger</th>
            <th style={{ width: 70, textAlign: 'right' }}>Steps</th>
            <th style={{ width: 80, textAlign: 'right' }}>Runs</th>
            <th style={{ width: 90, textAlign: 'right' }}>Success</th>
            <th style={{ width: 90 }}></th>
          </tr>
        </thead>
        <tbody>
          {items.map((p) => {
            const trig = trigger(p)
            const TrigIcon =
              trig.kind === 'cron' ? Clock : trig.kind === 'webhook' ? TriggerZap : Hand
            const runs = p.use_count ?? 0
            const success = p.success_rate ?? null
            return (
              <tr
                key={String(p.template_id || p.id)}
                onClick={() => onRun(p)}
              >
                <td>
                  <span
                    style={{
                      width: 24,
                      height: 24,
                      borderRadius: 5,
                      background: 'var(--bg-2)',
                      color: 'var(--fg)',
                      display: 'inline-flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                    }}
                  >
                    <BookMarked style={{ width: 12, height: 12 }} />
                  </span>
                </td>
                <td>
                  <div className="nm">{p.name || 'Untitled'}</div>
                  {p.description && (
                    <div
                      style={{
                        fontSize: 10.5,
                        color: 'var(--fg-3)',
                        marginTop: 2,
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                        maxWidth: 460,
                      }}
                    >
                      {p.description}
                    </div>
                  )}
                </td>
                <td>
                  <span className="trigger">
                    <TrigIcon style={{ width: 10, height: 10 }} />
                    {trig.label}
                  </span>
                </td>
                <td className="mono" style={{ textAlign: 'right' }}>
                  {p.steps?.length ?? 0}
                </td>
                <td className="mono" style={{ textAlign: 'right' }}>
                  {runs}
                </td>
                <td
                  className="mono"
                  style={{
                    textAlign: 'right',
                    color:
                      success == null
                        ? 'var(--fg-3)'
                        : success >= 99
                        ? 'hsl(82 50% 22%)'
                        : success >= 95
                        ? 'hsl(38 78% 27%)'
                        : 'hsl(var(--accent))',
                  }}
                >
                  {success == null ? '—' : `${Math.round(success)}%`}
                </td>
                <td>
                  <button
                    type="button"
                    className="l-btn l-btn--p"
                    style={{ height: 26, fontSize: 11.5, padding: '0 12px' }}
                    onClick={(e) => {
                      e.stopPropagation()
                      onRun(p)
                    }}
                  >
                    Run
                  </button>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
