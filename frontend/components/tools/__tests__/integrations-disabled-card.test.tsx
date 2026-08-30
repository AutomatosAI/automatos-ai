// PRD-233 S2 — the Tools page must not show an empty grid as if nothing
// exists: without a Composio key it says integrations are disabled, why, and
// how to fix it. Pure component test — no hooks, no network.

import { describe, it, expect, afterEach } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'
import React from 'react'

import {
  IntegrationsDisabledCard,
  INTEGRATIONS_DISABLED_NO_KEY_TITLE,
  INTEGRATIONS_DISABLED_TITLE,
  CATALOGUE_SYNCING_TITLE,
  CATALOGUE_SYNC_FAILED_TITLE,
} from '../integrations-disabled-card'
import type { IntegrationsStatus } from '@/hooks/use-tools-api'

afterEach(() => cleanup())

const noKey: IntegrationsStatus = {
  available: false,
  reason: 'COMPOSIO_API_KEY is not configured',
  key_configured: false,
  apps_cached: 0,
  last_sync: null,
  sync_status: null,
}

const keyed = { available: true, reason: null, key_configured: true }

describe('PRD-233 S2 — IntegrationsDisabledCard', () => {
  it('renders the honest disabled state with the fix and the reason when there is no key', () => {
    render(<IntegrationsDisabledCard status={noKey} />)
    const card = screen.getByTestId('integrations-disabled')
    expect(card).toHaveAttribute('role', 'alert')
    expect(screen.getByText(INTEGRATIONS_DISABLED_NO_KEY_TITLE)).toBeInTheDocument()
    const text = card.textContent || ''
    expect(text).toContain(
      'Add COMPOSIO_API_KEY to .env, then docker compose up -d backend; the catalogue syncs automatically on boot.',
    )
    expect(text).toContain('Reason: COMPOSIO_API_KEY is not configured')
    expect(text).toContain('Native platform tools keep working')
  })

  it('names the generic title (and the reason) when a key is set but the SDK is missing', () => {
    render(
      <IntegrationsDisabledCard
        status={{ ...noKey, key_configured: true, reason: 'Composio SDK is not installed (composio-core package missing)' }}
      />,
    )
    expect(screen.getByText(INTEGRATIONS_DISABLED_TITLE)).toBeInTheDocument()
    expect(screen.queryByText(INTEGRATIONS_DISABLED_NO_KEY_TITLE)).not.toBeInTheDocument()
    expect(screen.getByTestId('integrations-disabled').textContent).toContain('composio-core package missing')
  })

  it('renders nothing when integrations are available', () => {
    const { container } = render(
      <IntegrationsDisabledCard
        status={{ ...keyed, apps_cached: 880, last_sync: '2026-08-29T00:00:00Z', sync_status: 'completed' }}
      />,
    )
    expect(container).toBeEmptyDOMElement()
    expect(screen.queryByTestId('integrations-disabled')).not.toBeInTheDocument()
  })

  it('renders nothing while the status is still loading', () => {
    const { container } = render(<IntegrationsDisabledCard status={undefined} />)
    expect(container).toBeEmptyDOMElement()
  })

  it('says the catalogue is syncing on a keyed boot with an empty cache', () => {
    render(<IntegrationsDisabledCard status={{ ...keyed, apps_cached: 0, last_sync: null, sync_status: 'running' }} />)
    expect(screen.getByTestId('integrations-syncing')).toBeInTheDocument()
    expect(screen.getByText(CATALOGUE_SYNCING_TITLE)).toBeInTheDocument()
    expect(screen.queryByTestId('integrations-disabled')).not.toBeInTheDocument()
  })

  it('says the last sync failed instead of pretending to sync forever', () => {
    render(<IntegrationsDisabledCard status={{ ...keyed, apps_cached: 0, last_sync: null, sync_status: 'failed' }} />)
    expect(screen.getByTestId('integrations-sync-failed')).toBeInTheDocument()
    expect(screen.getByText(CATALOGUE_SYNC_FAILED_TITLE)).toBeInTheDocument()
  })
})
