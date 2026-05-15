'use client'

/**
 * Studio Missions queue (standalone /missions route).
 * Wraps MissionsBody with the editorial top and "New mission" actions.
 */

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import dynamic from 'next/dynamic'
import { Download, Play, Plus } from 'lucide-react'

import { useMissions } from '@/hooks/use-missions-api'
import { useWorkflowPlaybooks } from '@/hooks/use-playbook-api'

import { MissionsBody } from './missions-body'

const CreateMissionModal = dynamic(
  () => import('@/components/missions/create-mission-modal').then((m) => m.CreateMissionModal),
  { ssr: false },
)

export function StudioMissionsPage() {
  const router = useRouter()
  const [missionOpen, setMissionOpen] = useState(false)

  const { data: missionData } = useMissions({ limit: 100 })
  const { data: playbookData } = useWorkflowPlaybooks({ limit: 1 })

  const allMissions = missionData?.missions ?? []
  const totalPlaybooks =
    (playbookData as { total?: number; items?: unknown[] } | undefined)?.total ??
    (playbookData as { items?: unknown[] } | undefined)?.items?.length ??
    0

  const totals = {
    all: allMissions.length,
    needsApproval: allMissions.filter(
      (m) => m.state === 'awaiting_approval' || m.state === 'awaiting_human',
    ).length,
    running: allMissions.filter(
      (m) =>
        m.state === 'running' ||
        m.state === 'planning' ||
        m.state === 'verifying' ||
        m.state === 'paused',
    ).length,
  }

  return (
    <div className="cc-page">
      {/* Editorial top */}
      <div className="cc-headrow">
        <div className="cc-head">
          <p className="cc-eyebrow">The work · everything you’ve started</p>
          <h1 className="cc-h1">Missions</h1>
          <p className="cc-sub">
            <b>{totals.all} missions</b> ·{' '}
            <b>{totals.needsApproval} need your call</b> ·{' '}
            <b>{totals.running} running</b> · grouped by status, sorted by last
            activity.
          </p>
        </div>
        <div className="cc-actions">
          <button type="button" className="cc-btn" title="Export">
            <Download style={{ width: 12, height: 12 }} />
            Export
          </button>
          <button type="button" className="cc-btn" title="Replay">
            <Play style={{ width: 12, height: 12 }} />
            Replay
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

      {/* Sub-tabs (Playbooks · Missions) */}
      <nav className="cc-tabs" aria-label="Assignments sections">
        <button
          type="button"
          className="cc-tab"
          onClick={() => router.push('/playbooks' as any)}
        >
          <span>Playbooks</span>
          {totalPlaybooks > 0 && <span className="cc-tab-ct">{totalPlaybooks}</span>}
        </button>
        <button type="button" className="cc-tab active" aria-current="page">
          <span>Missions</span>
          {totals.all > 0 && <span className="cc-tab-ct">{totals.all}</span>}
        </button>
      </nav>

      <MissionsBody />

      <CreateMissionModal open={missionOpen} onOpenChange={setMissionOpen} />
    </div>
  )
}
