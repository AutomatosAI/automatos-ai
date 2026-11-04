/**
 * PRD-233 S6/S7 — the Settings tabs per edition.
 *
 * local → only what exists locally: Profile (S6), Session mode (PRD-234 S4),
 *         System Settings, Orchestrator, API Keys, Credentials, Channels (Telegram polls — no
 *         public URL needed), Notifications, Socials (PRD-251B US-B106: the workspace switch),
 *         Voice (the workspace's Auto Live switch), Appearance (PRD-261 A-07d: Themes, both
 *         editions), Phones & devices (PRD-266 US-005c, both editions). Webhooks / Widget SDK
 *         (hosted-edition surfaces) are hidden by the explicit allowlist, not by
 *         role — the local operator is super_admin.
 * saas  → the eight tabs as before plus Phones & devices, Voice, Socials, Appearance and Billing
 *         (PRD-265 US-011, hosted only); no Profile tab (Clerk owns it).
 *
 * Tab bodies are stubbed — this is about WHICH tabs exist, not their content.
 */
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen } from '@testing-library/react'

vi.mock('next/link', () => ({
  default: ({ href, children }: any) => <a href={href}>{children}</a>,
}))

// PRD-265 US-011: the open tab is read from `?tab=` (none here: the default tab).
vi.mock('next/navigation', () => ({
  useRouter: () => ({ replace: vi.fn(), push: vi.fn() }),
  useSearchParams: () => new URLSearchParams(''),
  usePathname: () => '/settings',
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
vi.mock('../WorkspaceSocialsCard', () => ({ WorkspaceSocialsCard: () => null }))  // PRD-251B US-B106
vi.mock('../WorkspaceVoiceLiveCard', () => ({ WorkspaceVoiceLiveCard: () => null }))  // the workspace's Auto Live switch
vi.mock('../themes/appearance-tab', () => ({ AppearanceTab: () => null }))  // PRD-261 A-07d
vi.mock('../phones-devices/phones-devices-tab', () => ({ PhonesDevicesTab: () => null }))  // PRD-266 US-005c
vi.mock('../billing/billing-tab', () => ({ BillingTab: () => null }))  // PRD-265 US-011 (hosted only)
// PRD-270 US-270-05b: the Automatos Cloud tab's own gate is automatos-cloud-panel.test.tsx's
vi.mock('../automatos-cloud-panel', () => ({ AutomatosCloudPanel: () => null, useHasAutomatosCloudTab: () => false }))
vi.mock('../what-auto-depends-on', () => ({ WhatAutoDependsOn: () => null }))  // PRD-268 US-W05
vi.mock('../proactive-auto/proactive-auto-section', () => ({ ProactiveAutoSection: () => null }))  // PRD-268 §6.9

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
      'Channels',
      'Notifications',
      'Phones & devices',
      'Voice',
      'Socials',
      'Appearance',
    ])
    expect(screen.getByRole('link', { name: /your profile/i })).toHaveAttribute('href', '/settings/profile')
    for (const hidden of ['webhooks', 'widget-sdk', 'billing']) {
      expect(LOCAL_EDITION_SETTINGS_TABS.has(hidden)).toBe(false)
    }
    // 2026-09-17: Channels is local — Telegram long-polls, so no public URL is needed
    expect(LOCAL_EDITION_SETTINGS_TABS.has('channels')).toBe(true)
    // PRD-234 S4: session mode is a local-only surface (the CLI host lane)
    expect(LOCAL_EDITION_SETTINGS_TABS.has('session-mode')).toBe(true)
    // PRD-251B US-B106: the workspace's Socials switch exists in both editions
    expect(LOCAL_EDITION_SETTINGS_TABS.has('socials')).toBe(true)
    // The workspace's Auto Live switch exists in both editions too
    expect(LOCAL_EDITION_SETTINGS_TABS.has('voice')).toBe(true)
    // PRD-261 A-07d: Appearance › Themes exists in both editions
    expect(LOCAL_EDITION_SETTINGS_TABS.has('appearance')).toBe(true)
    // PRD-266 US-005c: Phones & devices exists in both editions
    expect(LOCAL_EDITION_SETTINGS_TABS.has('phones-devices')).toBe(true)
  })

  it('saas: the eight tabs render as before, plus Phones & devices, Voice, Socials, Appearance and Billing, and there is no Profile tab', async () => {
    const { SettingsPanel } = await loadPanel('saas')
    render(<SettingsPanel />)

    expect(tabLabels()).toEqual([
      'System Settings',
      'Orchestrator',
      'Webhooks',
      'API Keys',
      'Billing',
      'Credentials',
      'Channels',
      'Notifications',
      'Phones & devices',
      'Voice',
      'Socials',
      'Appearance',
      'Widget SDK',
    ])
    expect(screen.queryByRole('link', { name: /your profile/i })).not.toBeInTheDocument()
  })
})
