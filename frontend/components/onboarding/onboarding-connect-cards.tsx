'use client'

import { useConnectedApps } from '@/hooks/use-composio-api'
import { useWorkspace } from '@/components/workspace-provider'
import { ConnectAppCard } from './connect-app-card'

/**
 * PRD-222 US-019 (W2·S3) — the chat-surface mount for inline connect cards.
 *
 * During the build stage of onboarding (`onboarding.stage === 'building'`), Auto
 * "requests the 1–2 app connections inline" (see the OnboardingSection building
 * block). This surfaces a {@link ConnectAppCard} in chat for each app the
 * workspace has been asked to connect but hasn't finished — i.e. every Composio
 * connection NOT yet `active` (a `pending`/`added` row). Those rows are the real,
 * existing signal of "an app Auto asked for": no new state, route, or event bus,
 * and no dependency on the plan/proposal machinery (the Q1-gated W2·S1/S2 kit,
 * out of this wave). It renders nothing when there is nothing mid-connect.
 */
export function OnboardingConnectCards({ className = '' }: { className?: string }) {
  const { workspace } = useWorkspace()
  const isBuilding = workspace?.onboarding?.stage === 'building'
  const { data: connections = [] } = useConnectedApps({ enabled: isBuilding })

  if (!isBuilding) return null

  // Apps requested but not yet connected — the ones still needing the user.
  const seen = new Set<string>()
  const pending = connections.filter((c) => {
    const app = (c.app_name || '').toUpperCase()
    if (!app || c.status === 'active' || seen.has(app)) return false
    seen.add(app)
    return true
  })

  if (pending.length === 0) return null

  return (
    <div
      data-testid="onboarding-connect-cards"
      className={['space-y-2', className].filter(Boolean).join(' ')}
    >
      {pending.map((c) => (
        <ConnectAppCard key={c.app_name} appName={c.app_name} />
      ))}
    </div>
  )
}
