import { describe, it, expect } from 'vitest'
import { readFileSync } from 'fs'
import path from 'path'

import { linkFor } from '@/components/notifications/notification-bell'
import type { NotificationRow } from '@/hooks/use-notifications-api'

// PRD-227 US-003 — bell deep-link drift guard.
//
// Every `link_type` the backend writes to a NOTIFICATION ROW must resolve to a
// route in linkFor(), so a future producer added without a bell case fails CI
// here instead of silently marking-read and navigating nowhere (PRD G4: every
// notification is navigable).
//
// FIXTURE = the link_types passed to NotificationDispatcher.dispatch(...) — the
// only writer of notification rows the bell renders. Regenerate from the
// orchestrator repo root with:
//
//   grep -rhoE 'link_type=[\"'"'"'][a-z_]+[\"'"'"']' orchestrator --include='*.py' \
//     | sort -u   # then keep only the NotificationDispatcher.dispatch call sites
//
// Verified 2026-08-27. Note the reconciliation vs the PRD's provisional list:
//   • 'scheduled_task' is written ONLY to chat-message sources
//     (deliver_background_message), never a notification row, and no frontend
//     navigates message.source.link_type — so it is out of the bell's scope.
//   • 'trigger' / 'agent' have NO backend producer (linkFor keeps defensive
//     cases, but they are deliberately NOT in this fixture so removing those
//     dead cases later is not blocked by this guard).
const BACKEND_NOTIFICATION_LINK_TYPES = [
  'task', // api/board_tasks.py, services/board_dispatcher.py
  'mission', // services/coordinator_service.py (_dispatch_mission_event)
  'playbook', // api/recipe_executor.py, services/playbook_scheduler.py
  'heartbeat', // services/heartbeat_service.py
  'report', // services/report_service.py
  'approval_grant', // modules/tools/execution/tool_grants.py, services/board_approval.py, services/watch_rerun.py
  'watch', // services/watch_notifications.py
  'question', // PRD-225: modules/tools/discovery/handlers_asks.py (question_pending)
] as const

function row(link_type: string | null, link_id: string | null = null): NotificationRow {
  return {
    id: 'n1',
    workspace_id: 'ws1',
    user_id: 1,
    event_type: 'x',
    title: 't',
    message: null,
    link_type,
    link_id,
    agent_id: null,
    agent_name: null,
    status: 'ok',
    read_at: null,
    dismissed_at: null,
    created_at: '2026-08-27T00:00:00Z',
  }
}

describe('PRD-227 US-003 — bell deep-link drift guard', () => {
  it('routes every backend notification link_type to a non-null destination', () => {
    for (const lt of BACKEND_NOTIFICATION_LINK_TYPES) {
      const route = linkFor(row(lt))
      expect(route, `link_type '${lt}' has no linkFor route — add a case`).not.toBeNull()
      expect(typeof route).toBe('string')
      expect(route!.length).toBeGreaterThan(0)
    }
  })

  it('routes approval_grant → Governance and watch → Watchlist (the new cases)', () => {
    expect(linkFor(row('approval_grant'))).toBe('/command-center?tab=governance')
    expect(linkFor(row('watch'))).toBe('/command-center?tab=watchlist')
  })

  it('routes question → the Questions tab (PRD-225)', () => {
    expect(linkFor(row('question'))).toBe('/command-center?tab=questions')
  })

  it('the new routes use tab params the Command Center shell actually reads', () => {
    // Cross-file guard: a tab rename in the shell must break this, not just
    // dead-link the bell. Extract each new route's ?tab= value and assert the
    // shell declares it as a real tab key.
    const shell = readFileSync(
      path.resolve(__dirname, '..', '..', 'command-center', 'command-center-shell.tsx'),
      'utf8',
    )
    for (const lt of ['approval_grant', 'watch', 'question'] as const) {
      const route = linkFor(row(lt))!
      const tab = new URLSearchParams(route.split('?')[1]).get('tab')!
      expect(shell).toContain(`'${tab}'`)
    }
  })

  // --- self-check: prove the guard can actually fail -----------------------

  it('returns null for an unhandled link_type (so the guard above is not vacuous)', () => {
    // If any fixture literal lost its case it would fall to this default and
    // return null, failing the first test. This proves that path is live.
    expect(linkFor(row('definitely_not_a_link_type'))).toBeNull()
  })

  it('returns null when link_type is absent', () => {
    expect(linkFor(row(null))).toBeNull()
  })
})
