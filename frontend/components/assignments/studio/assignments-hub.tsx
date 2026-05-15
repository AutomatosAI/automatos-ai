'use client'

/**
 * Studio Assignments hub.
 *
 * Editorial top → "Start something" 4-up EntryGrid → Needs approval
 * cards (conditional) → Playbooks/Missions flip tabs (URL-synced via
 * ?tab=) → the matching Body component (PlaybooksBody / MissionsBody).
 *
 * The flip tabs replace CD round 4's "Most-used playbooks" + 5-card
 * "Marketplace" lower strips — the user wants to browse one or the
 * other in place, not see three stacked teaser sections.
 */

import { useCallback, useMemo, useState } from 'react'
import { useRouter, useSearchParams, usePathname } from 'next/navigation'
import { Filter, Plus } from 'lucide-react'
import dynamic from 'next/dynamic'

import { useMissions } from '@/hooks/use-missions-api'
import { useWorkflowPlaybooks } from '@/hooks/use-playbook-api'

import { EntryGrid, type EntryType } from './entry-grid'
import { StatusHead } from './status-head'
import { MissionCard } from './mission-card'
import { MissionsBody } from './missions-body'
import { PlaybooksBody } from './playbooks-body'

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

type Tab = 'playbooks' | 'missions'

function isAwaitingApproval(m: MissionResponse): boolean {
  return m.state === 'awaiting_approval' || m.state === 'awaiting_human'
}

export function StudioAssignmentsHub() {
  const router = useRouter()
  const pathname = usePathname()
  const searchParams = useSearchParams()

  const tabParam = (searchParams?.get('tab') as Tab | null) ?? null
  const tab: Tab = tabParam === 'missions' || tabParam === 'playbooks' ? tabParam : 'playbooks'

  const [missionOpen, setMissionOpen] = useState(false)
  const [playbookOpen, setPlaybookOpen] = useState(false)
  const [taskOpen, setTaskOpen] = useState(false)

  const { data: missionData } = useMissions({ limit: 30 })
  const { data: playbookData } = useWorkflowPlaybooks({ limit: 1 })

  const missions = missionData?.missions ?? []
  const needsApproval = useMemo(
    () => missions.filter(isAwaitingApproval).slice(0, 4),
    [missions],
  )
  const totalPlaybooks =
    (playbookData as { total?: number; items?: unknown[] } | undefined)?.total ??
    (playbookData as { items?: unknown[] } | undefined)?.items?.length ??
    0

  const setTab = useCallback(
    (next: Tab) => {
      const params = new URLSearchParams(searchParams?.toString() ?? '')
      params.set('tab', next)
      router.push(`${pathname}?${params.toString()}` as any, { scroll: false })
    },
    [pathname, router, searchParams],
  )

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

      {/* Flip-tabs: Playbooks · Missions */}
      <nav className="cc-tabs" aria-label="Assignments sections">
        <button
          type="button"
          className={`cc-tab${tab === 'playbooks' ? ' active' : ''}`}
          aria-current={tab === 'playbooks' ? 'page' : undefined}
          onClick={() => setTab('playbooks')}
        >
          <span>Playbooks</span>
          {totalPlaybooks > 0 && <span className="cc-tab-ct">{totalPlaybooks}</span>}
        </button>
        <button
          type="button"
          className={`cc-tab${tab === 'missions' ? ' active' : ''}`}
          aria-current={tab === 'missions' ? 'page' : undefined}
          onClick={() => setTab('missions')}
        >
          <span>Missions</span>
          {missions.length > 0 && <span className="cc-tab-ct">{missions.length}</span>}
        </button>
      </nav>

      {/* Tab body */}
      {tab === 'playbooks' ? <PlaybooksBody /> : <MissionsBody />}

      <CreateMissionModal open={missionOpen} onOpenChange={setMissionOpen} />
      <CreatePlaybookModal
        open={playbookOpen}
        onClose={() => setPlaybookOpen(false)}
      />
      <CreateTaskDialog open={taskOpen} onOpenChange={setTaskOpen} />
    </div>
  )
}
