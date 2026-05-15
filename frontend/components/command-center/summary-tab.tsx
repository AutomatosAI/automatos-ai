'use client'

/**
 * SummaryTab — CD round-4 SUM-A (editorial, attention-led).
 *
 * Two-column layout on wide screens, single column under 1200px. Left column
 * leads with "Needs your eyes" + roster; right column carries Schedule, Cost,
 * Approvals. Every widget is wired to a real hook and renders an editorial
 * empty state when data is absent — no fabricated values.
 */

import Link from 'next/link'
import { AttentionList } from './widgets/attention-list'
import { AgentRoster } from './widgets/agent-roster'
import { ScheduleMini } from './widgets/schedule-mini'
import { CostBlock } from './widgets/cost-block'
import { ApprovalsList } from './widgets/approvals-list'
import { useDecisionsNeeded, useApprovalGates } from '@/hooks/use-kpi-api'
import { useAgents } from '@/hooks/use-agent-api'
import { useActivityStats } from '@/hooks/use-activity-api'

interface PanelHeadProps {
  title: string
  meta?: string
  right?: React.ReactNode
}

function PanelHead({ title, meta, right }: PanelHeadProps) {
  return (
    <div className="cc-panel-head">
      <span className="t">{title}</span>
      {meta && <span className="meta">{meta}</span>}
      {right && <div className="right">{right}</div>}
    </div>
  )
}

export function SummaryTab() {
  const { data: decisions } = useDecisionsNeeded(10)
  const { data: approvals } = useApprovalGates('30d')
  const { data: agents } = useAgents()
  const { data: stats } = useActivityStats('1d')

  const decisionsTotal = decisions?.total ?? 0
  const approvalsCount = approvals?.pending_count ?? 0
  const agentsTotal = Array.isArray(agents) ? agents.length : 0
  const agentsActive = stats?.agents_active ?? 0
  const agentsIdle = Math.max(agentsTotal - agentsActive, 0)

  return (
    <div className="cc-sum-grid">
      {/* Left column */}
      <div className="cc-sum-col">
        <div className="cc-panel">
          <PanelHead
            title="Needs your eyes"
            meta={decisionsTotal > 0 ? `${decisionsTotal} item${decisionsTotal === 1 ? '' : 's'}` : 'all clear'}
            right={
              <Link
                href={'/command-center?tab=activity&status=err' as any}
                className="cc-btn"
                style={{ height: 26, fontSize: 11.5, padding: '0 10px' }}
              >
                View all →
              </Link>
            }
          />
          <AttentionList limit={4} />
        </div>

        <div className="cc-panel">
          <PanelHead
            title="Agent roster"
            meta={
              agentsTotal > 0
                ? `${agentsTotal} agents · ${agentsActive} active · ${agentsIdle} idle`
                : 'no agents yet'
            }
            right={
              <Link
                href={'/agents' as any}
                className="cc-btn"
                style={{ height: 26, fontSize: 11.5, padding: '0 10px' }}
              >
                Roster →
              </Link>
            }
          />
          <AgentRoster limit={8} />
        </div>
      </div>

      {/* Right column */}
      <div className="cc-sum-col">
        <div className="cc-panel">
          <PanelHead
            title="Schedule"
            meta="this week"
            right={
              <Link
                href={'/command-center?tab=calendar' as any}
                className="cc-btn"
                style={{ height: 26, fontSize: 11.5, padding: '0 10px' }}
              >
                Calendar →
              </Link>
            }
          />
          <ScheduleMini limit={6} />
        </div>

        <div className="cc-panel">
          <PanelHead title="Cost tracker" meta="1d spend" />
          <CostBlock />
        </div>

        <div className="cc-panel">
          <PanelHead
            title="Approval gates"
            meta={approvalsCount > 0 ? `${approvalsCount} pending` : 'all clear'}
            right={
              approvalsCount > 0 ? (
                <Link
                  href={'/missions?state=awaiting_approval' as any}
                  className="cc-btn"
                  style={{ height: 26, fontSize: 11.5, padding: '0 10px' }}
                >
                  Open queue →
                </Link>
              ) : undefined
            }
          />
          <ApprovalsList limit={3} />
        </div>
      </div>
    </div>
  )
}
