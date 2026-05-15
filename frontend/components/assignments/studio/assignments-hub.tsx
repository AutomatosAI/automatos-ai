'use client'

/**
 * Studio Assignments hub — CD round 4 HUB-A ("launcher first").
 *
 *  1. Editorial top: eyebrow · h1 · sub · Filter/New mission actions
 *  2. Start something — EntryGrid (Mission · Playbook · Plan · Task)
 *  3. Needs approval — conditional MissionCard grid (warn-bordered)
 *  4. Running now — MissionCard grid (3-up)
 *  5. Most-used playbooks — PlaybookCard grid (3-up)
 *  6. Recommended in the marketplace — MktCard strip (5-up)
 *
 * Mission/playbook clicks route to existing detail pages; Plan opens
 * /chat?mode=plan; Marketplace cards route to /marketplace?id=. CTAs
 * never invent new flows — they always re-use what we have.
 */

import { useCallback, useMemo, useState } from 'react'
import { useRouter } from 'next/navigation'
import { Filter, Plus } from 'lucide-react'
import dynamic from 'next/dynamic'

import { useMissions } from '@/hooks/use-missions-api'
import { useWorkflowPlaybooks } from '@/hooks/use-playbook-api'
import {
  useRecommendedAssignments,
  type RecommendedItem,
} from '@/hooks/use-assignments-api'

import { EntryGrid, type EntryType } from './entry-grid'
import { StatusHead } from './status-head'
import { MissionCard } from './mission-card'
import { PlaybookCard, type PlaybookLike } from './playbook-card'
import { MktCard } from './mkt-card'

import type { MissionResponse } from '@/types/missions'

const CreateMissionModal = dynamic(
  () => import('@/components/missions/create-mission-modal').then((m) => m.CreateMissionModal),
  { ssr: false },
)
const CreatePlaybookModal = dynamic(
  () => import('@/components/workflows/create-playbook-modal').then((m) => m.CreatePlaybookModal),
  { ssr: false },
)
const CreateTaskDialog = dynamic(
  () => import('@/components/activity/board/create-task-dialog').then((m) => m.CreateTaskDialog),
  { ssr: false },
)

function isAwaitingApproval(m: MissionResponse): boolean {
  return m.state === 'awaiting_approval' || m.state === 'awaiting_human'
}

function isRunning(m: MissionResponse): boolean {
  return (
    m.state === 'running' ||
    m.state === 'planning' ||
    m.state === 'verifying' ||
    m.state === 'paused'
  )
}

export function StudioAssignmentsHub() {
  const router = useRouter()

  const [missionOpen, setMissionOpen] = useState(false)
  const [playbookOpen, setPlaybookOpen] = useState(false)
  const [taskOpen, setTaskOpen] = useState(false)

  const { data: missionData } = useMissions({ limit: 30 })
  const { data: playbookData } = useWorkflowPlaybooks({ limit: 24, sort_by: 'use_count' })
  const { data: recommended } = useRecommendedAssignments(8)

  const missions = missionData?.missions ?? []
  const needsApproval = useMemo(() => missions.filter(isAwaitingApproval).slice(0, 4), [missions])
  const running = useMemo(() => missions.filter(isRunning).slice(0, 3), [missions])

  const playbooks = (playbookData as { items?: PlaybookLike[] } | undefined)?.items ?? []
  const mostUsed = useMemo(
    () =>
      [...playbooks]
        .sort((a, b) => (b.use_count ?? 0) - (a.use_count ?? 0))
        .slice(0, 3),
    [playbooks],
  )

  const marketplace = (recommended?.items ?? []).slice(0, 5)

  const handleEntry = useCallback(
    (type: EntryType) => {
      if (type === 'mission') setMissionOpen(true)
      else if (type === 'playbook') setPlaybookOpen(true)
      else if (type === 'task') setTaskOpen(true)
      else if (type === 'plan') router.push('/chat?mode=plan&from=assignments')
    },
    [router],
  )

  const handleMission = useCallback(
    (id: string) => router.push(`/missions/${id}` as any),
    [router],
  )
  const handlePlaybook = useCallback(
    (p: PlaybookLike) => {
      const id = p.template_id || (p.id != null ? String(p.id) : '')
      if (id) router.push(`/playbooks?id=${encodeURIComponent(id)}` as any)
    },
    [router],
  )
  const handleMarketplace = useCallback(
    (item: RecommendedItem) => {
      if (item.source === 'marketplace') {
        router.push(`/marketplace?id=${item.id}` as any)
      } else if (item.type === 'mission') {
        router.push(`/missions/${item.id}` as any)
      } else {
        router.push(`/playbooks?id=${item.id}` as any)
      }
    },
    [router],
  )

  return (
    <div className="cc-page">
      {/* Editorial top */}
      <div className="cc-headrow">
        <div className="cc-head">
          <p className="cc-eyebrow">The work · pilot · {missions.length} op</p>
          <h1 className="cc-h1">Assignments</h1>
          <p className="cc-sub">
            Plan, schedule, and orchestrate work for your crew. Start something
            new — or pick up what’s running.
          </p>
        </div>
        <div className="cc-actions">
          <button type="button" className="cc-btn" title="Filter">
            <Filter style={{ width: 12, height: 12 }} />
            Filter
          </button>
          <button
            type="button"
            className="cc-btn primary"
            onClick={() => setMissionOpen(true)}
          >
            <Plus style={{ width: 12, height: 12 }} />
            New mission
          </button>
        </div>
      </div>

      {/* Section 1 — Start something */}
      <section style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 10 }}>
          <span
            style={{
              fontFamily: 'var(--serif)',
              fontSize: 17,
              fontWeight: 500,
              color: 'var(--fg)',
            }}
          >
            Start something
          </span>
          <span
            style={{
              fontFamily: 'var(--font-geist-mono, monospace)',
              fontSize: 10,
              color: 'var(--fg-3)',
              letterSpacing: '0.06em',
            }}
          >
            pick the kind of work · all share the same composer
          </span>
        </div>
        <EntryGrid recommended="mission" onPick={handleEntry} />
      </section>

      {/* Section 2 — Needs approval (only when present) */}
      {needsApproval.length > 0 && (
        <section>
          <StatusHead
            pip="hsl(38 78% 50%)"
            label="Needs approval"
            count={`${needsApproval.length} waiting`}
            right={
              <button
                type="button"
                className="l-btn"
                style={{ height: 26, fontSize: 11.5, padding: '0 10px' }}
                onClick={() => router.push('/missions?state=awaiting_approval' as any)}
              >
                Open queue →
              </button>
            }
          />
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(2, 1fr)',
              gap: 12,
            }}
          >
            {needsApproval.map((m) => (
              <MissionCard key={m.id} mission={m} onOpen={handleMission} />
            ))}
          </div>
        </section>
      )}

      {/* Section 3 — Running */}
      <section>
        <StatusHead
          pip="hsl(var(--foreground))"
          label="Running now"
          count={`${running.length} active`}
          right={
            <button
              type="button"
              className="l-btn"
              style={{ height: 26, fontSize: 11.5, padding: '0 10px' }}
              onClick={() => router.push('/missions' as any)}
            >
              All missions →
            </button>
          }
        />
        {running.length > 0 ? (
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(3, 1fr)',
              gap: 12,
            }}
          >
            {running.map((m) => (
              <MissionCard key={m.id} mission={m} onOpen={handleMission} />
            ))}
          </div>
        ) : (
          <div
            className="pg-panel"
            style={{
              padding: 24,
              textAlign: 'center',
              color: 'var(--fg-3)',
              fontFamily: 'var(--font-geist-mono, monospace)',
              fontSize: 11.5,
              letterSpacing: '0.04em',
            }}
          >
            Nothing running right now. Start something above, or schedule a
            playbook to run when you’re away.
          </div>
        )}
      </section>

      {/* Section 4 — Most-used playbooks */}
      {mostUsed.length > 0 && (
        <section>
          <StatusHead
            pip="hsl(82 50% 22%)"
            label="Most-used playbooks"
            count={`${playbooks.length} in library`}
            right={
              <button
                type="button"
                className="l-btn"
                style={{ height: 26, fontSize: 11.5, padding: '0 10px' }}
                onClick={() => router.push('/playbooks' as any)}
              >
                Browse library →
              </button>
            }
          />
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(3, 1fr)',
              gap: 12,
            }}
          >
            {mostUsed.map((p) => (
              <PlaybookCard
                key={String(p.template_id || p.id)}
                playbook={p}
                onRun={handlePlaybook}
              />
            ))}
          </div>
        </section>
      )}

      {/* Section 5 — Marketplace */}
      {marketplace.length > 0 && (
        <section>
          <StatusHead
            pip="hsl(var(--info))"
            label="Recommended in the marketplace"
            count={`${marketplace.length} picks for your stack`}
            right={
              <button
                type="button"
                className="l-btn"
                style={{ height: 26, fontSize: 11.5, padding: '0 10px' }}
                onClick={() => router.push('/marketplace' as any)}
              >
                Browse marketplace →
              </button>
            }
          />
          <div className="mkt-strip" style={{ paddingBottom: 24 }}>
            {marketplace.map((item) => (
              <MktCard
                key={`${item.source}-${item.id}`}
                item={item}
                onClick={handleMarketplace}
              />
            ))}
          </div>
        </section>
      )}

      <CreateMissionModal open={missionOpen} onOpenChange={setMissionOpen} />
      <CreatePlaybookModal
        open={playbookOpen}
        onClose={() => setPlaybookOpen(false)}
      />
      <CreateTaskDialog open={taskOpen} onOpenChange={setTaskOpen} />
    </div>
  )
}
