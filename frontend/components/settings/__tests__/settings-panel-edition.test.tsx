/**
 * PRD-233 S6/S7 — the Settings tabs per edition.
 *
 * local → only what exists locally: Profile (S6), Session mode (PRD-234 S4),
 *         System Settings, Orchestrator, API Keys, Credentials, Notifications. Webhooks / Channels / Widget SDK
 *         (hosted-edition surfaces) are hidden by the explicit allowlist, not by
 *         role — the local operator is super_admin.
 * saas  → the eight tabs exactly as before; no Profile tab (Clerk owns it).
 *
 * Tab bodies are stubbed — this is about WHICH tabs exist, not their content.
 */
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen } from '@testing-library/react'

vi.mock('next/link', () => ({
  default: ({ href, children }: any) => <a href={href}>{children}</a>,
}))

vi.mock('@/contexts/role-context', () => ({
  useSystemRole: () => ({ isAdmin: true, systemRole: 'super_admin', isLoading: false }),
}))

vi.mock('@/components/shared', () => ({
  PageHeader: () => null,
  FilterTabs: ({ tabs, children }: any) => (
    <div>
      <nav aria-label="settings tabs">
        {tabs.map((tab: any) => (
          <button key={tab.value} type="button">
            {tab.label}
          </button>
        ))}
      </nav>
      {children}
    </div>
  ),
  TabsContent: ({ children }: any) => <section>{children}</section>,
}))

vi.mock('../CredentialsTab', () => ({ CredentialsTab: () => null }))
vi.mock('../SystemSettingsTab', () => ({ default: () => null }))
vi.mock('../SystemLLMSettingsTab', () => ({ default: () => null }))
vi.mock('../WebhooksSettingsTab', () => ({ default: () => null }))
vi.mock('../ApiKeysSettingsTab', () => ({ ApiKeysSettingsTab: () => null }))
vi.mock('../ChannelsSettingsTab', () => ({ ChannelsSettingsTab: () => null }))
vi.mock('../ApiKeyManager', () => ({ ApiKeyManager: () => null }))
vi.mock('../WidgetSdkTab', () => ({ WidgetSdkTab: () => null }))
vi.mock('../NotificationsSettingsTab', () => ({ NotificationsSettingsTab: () => null }))
vi.mock('../SessionModeTab', () => ({ SessionModeTab: () => null }))  // PRD-234 S4 (local only)

async function loadPanel(edition: 'local' | 'saas') {
  vi.doMock('@/lib/auth-edition', () => ({
    authEdition: edition,
    isLocal: edition === 'local',
    isSaaS: edition === 'saas',
  }))
  const mod = await import('../SettingsPanel')
  return mod
}

function tabLabels(): string[] {
  return screen.getAllByRole('button').map((button) => button.textContent ?? '')
}

afterEach(() => {
  vi.doUnmock('@/lib/auth-edition')
  vi.resetModules()
})

describe('SettingsPanel edition gating (PRD-233 S6/S7)', () => {
  it('local: shows Profile + the tabs that exist locally, hides hosted-only surfaces', async () => {
    const { SettingsPanel, LOCAL_EDITION_SETTINGS_TABS } = await loadPanel('local')
    render(<SettingsPanel />)

    expect(tabLabels()).toEqual([
      'Profile',
      'Session mode',
      'System Settings',
      'Orchestrator',
      'API Keys',
      'Credentials',
      'Notifications',
    ])
    expect(screen.getByRole('link', { name: /your profile/i })).toHaveAttribute('href', '/settings/profile')
    for (const hidden of ['webhooks', 'channels', 'widget-sdk']) {
      expect(LOCAL_EDITION_SETTINGS_TABS.has(hidden)).toBe(false)
    }
    // PRD-234 S4: session mode is a local-only surface (the CLI host lane)
    expect(LOCAL_EDITION_SETTINGS_TABS.has('session-mode')).toBe(true)
  })

  it('saas: the eight tabs render exactly as before and there is no Profile tab', async () => {
    const { SettingsPanel } = await loadPanel('saas')
    render(<SettingsPanel />)

    expect(tabLabels()).toEqual([
      'System Settings',
      'Orchestrator',
      'Webhooks',
      'API Keys',
      'Credentials',
      'Channels',
      'Notifications',
      'Widget SDK',
    ])
    expect(screen.queryByRole('link', { name: /your profile/i })).not.toBeInTheDocument()
  })
})
