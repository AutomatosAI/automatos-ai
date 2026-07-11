'use client'

/**
 * GovernanceTab — PRD-196 (P2-15). The single Command Centre pillar for
 * governance: Approvals · Audit · Policy · Compliance.
 *
 * The whole tab is workspace-admin-only (every endpoint it calls is gated by
 * the canonical `require_workspace_admin`, PRD-185 S12 / PRD-196 S2). This is
 * the human surface for the durable-grant machinery, the audit log, the policy
 * posture/budget, and GDPR self-service that were previously API-only.
 *
 * S1 lands the shell + the Approvals inbox (lead pane); S3 adds Audit, S4 adds
 * Policy, S7 adds Compliance — each a working pane, no dead UI.
 */

import { useState } from 'react'
import { ApprovalsInbox } from './governance/approvals-inbox'
import { AuditPane } from './governance/audit-pane'
import { PolicyPane } from './governance/policy-pane'

type PaneKey = 'approvals' | 'audit' | 'policy' | 'compliance'

interface Pane {
  key: PaneKey
  label: string
  node: React.ReactNode
}

// Panes register here; later stories append their entry (S3 audit, S4 policy,
// S7 compliance). Approvals is the lead pane.
const PANES: Pane[] = [
  { key: 'approvals', label: 'Approvals', node: <ApprovalsInbox /> },
  { key: 'audit', label: 'Audit', node: <AuditPane /> },
  { key: 'policy', label: 'Policy', node: <PolicyPane /> },
]

export function GovernanceTab() {
  const [pane, setPane] = useState<PaneKey>('approvals')
  const active = PANES.find((p) => p.key === pane) ?? PANES[0]

  return (
    <div className="flex flex-col gap-4">
      {PANES.length > 1 && (
        <nav className="cc-tabs" aria-label="Governance sections">
          {PANES.map((p) => (
            <button
              key={p.key}
              type="button"
              className={`cc-tab${p.key === active.key ? ' active' : ''}`}
              aria-current={p.key === active.key ? 'page' : undefined}
              onClick={() => setPane(p.key)}
            >
              <span>{p.label}</span>
            </button>
          ))}
        </nav>
      )}
      <div>{active.node}</div>
    </div>
  )
}
