'use client'

/**
 * Studio Assignments hub.
 *
 * Editorial top → "Start something" 4-up EntryGrid → Playbooks/Missions
 * flip tabs (URL-synced via ?tab=) → the matching Body component
 * (PlaybooksBody / MissionsBody).
 *
 * Awaiting-approval missions surface inside the Missions tab via the
 * "Needs your call" group in MissionsBody's Grouped view, so the hub
 * doesn't render a separate cards section above the tabs.
 */

import { useCallback, useState } from 'react'
import { useRouter, useSearchParams, usePathname } from 'next/navigation'
import dynamic from 'next/dynamic'

import { useMissions } from '@/hooks/use-missions-api'
import { useWorkflowPlaybooks } from '@/hooks/use-playbook-api'

import { EntryGrid, type EntryType } from './entry-grid'
import { MissionsBody } from './missions-body'
import { PlaybooksBody } from './playbooks-body'

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

  const missionCount = missionData?.missions?.length ?? 0
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

  return (
    <div className="cc-page">
      {/* Editorial top — actions removed: Filter was dead, and "+ New
          mission" duplicated the Mission card in the EntryGrid below. */}
      <div className="cc-headrow">
        <div className="cc-head">
          <p className="cc-eyebrow">The work · pilot · {missionCount} op</p>
          <h1 className="cc-h1">Assignments</h1>
          <p className="cc-sub">
            Plan, schedule, and orchestrate work for your crew. Start something
            new — or pick up what’s running.
          </p>
        </div>
      </div>

      {/* Flip tabs — moved above Start something so they're the first
          navigation element visible at scroll=0 and stay sticky as the
          user scrolls into the library below. */}
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
          {missionCount > 0 && <span className="cc-tab-ct">{missionCount}</span>}
        </button>
      </nav>

      {/* Body — scrolls. EntryGrid lives at the top of the body so it
          scrolls away as the user moves into the library; the editorial
          top + tabs above stay pinned. */}
      <div className="cc-body">
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

        {tab === 'playbooks' ? <PlaybooksBody /> : <MissionsBody />}
      </div>

      <CreateMissionModal open={missionOpen} onOpenChange={setMissionOpen} />
      <CreatePlaybookModal
        open={playbookOpen}
        onClose={() => setPlaybookOpen(false)}
      />
      <CreateTaskDialog open={taskOpen} onOpenChange={setTaskOpen} />
    </div>
  )
}
