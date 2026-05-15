'use client'

/**
 * Studio Playbooks library (standalone /playbooks route).
 * Wraps PlaybooksBody with the editorial top and create CTAs.
 */

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import dynamic from 'next/dynamic'
import { Store, Upload, Plus } from 'lucide-react'

import { useWorkflowPlaybooks } from '@/hooks/use-playbook-api'
import { useMissions } from '@/hooks/use-missions-api'

import { PlaybooksBody } from './playbooks-body'

const CreatePlaybookModal = dynamic(
  () => import('@/components/workflows/create-playbook-modal').then((m) => m.CreatePlaybookModal),
  { ssr: false },
)

export function StudioPlaybooksPage() {
  const router = useRouter()
  const [playbookOpen, setPlaybookOpen] = useState(false)

  const { data } = useWorkflowPlaybooks({ limit: 1 })
  const { data: missionData } = useMissions({ limit: 1 })

  const totalPlaybooks =
    (data as { total?: number; items?: unknown[] } | undefined)?.total ??
    (data as { items?: unknown[] } | undefined)?.items?.length ??
    0
  const missionCount = missionData?.total ?? missionData?.missions?.length ?? 0

  return (
    <div className="cc-page">
      <div className="cc-headrow">
        <div className="cc-head">
          <p className="cc-eyebrow">The work · reusable routines</p>
          <h1 className="cc-h1">Playbooks</h1>
          <p className="cc-sub">
            Reusable routines. Schedule them, trigger them, fork them, version
            them. Save anything that works twice.
          </p>
        </div>
        <div className="cc-actions">
          <button
            type="button"
            className="cc-btn"
            onClick={() => router.push('/marketplace?tab=playbooks' as any)}
          >
            <Store style={{ width: 12, height: 12 }} />
            Marketplace
          </button>
          <button type="button" className="cc-btn" title="Import">
            <Upload style={{ width: 12, height: 12 }} />
            Import
          </button>
          <button
            type="button"
            className="cc-btn primary"
            onClick={() => setPlaybookOpen(true)}
          >
            <Plus style={{ width: 12, height: 12 }} />
            New playbook
          </button>
        </div>
      </div>

      <nav className="cc-tabs" aria-label="Assignments sections">
        <button type="button" className="cc-tab active" aria-current="page">
          <span>Playbooks</span>
          {totalPlaybooks > 0 && <span className="cc-tab-ct">{totalPlaybooks}</span>}
        </button>
        <button
          type="button"
          className="cc-tab"
          onClick={() => router.push('/missions' as any)}
        >
          <span>Missions</span>
          {missionCount > 0 && <span className="cc-tab-ct">{missionCount}</span>}
        </button>
      </nav>

      <PlaybooksBody />

      <CreatePlaybookModal
        open={playbookOpen}
        onClose={() => setPlaybookOpen(false)}
      />
    </div>
  )
}
