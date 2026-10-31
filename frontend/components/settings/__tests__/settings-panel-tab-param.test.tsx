/**
 * PRD-265 US-011 — Settings reads and writes `?tab=`.
 *
 * `/settings?tab=billing` (the paused banner's "Top up", Auto's onboarding link
 * `/settings?tab=billing&plan=pro`, Stripe's return pages) opens the Billing tab; a tab the
 * address names that this person does not have falls back to the default; choosing a tab
 * switches at once and rewrites `?tab=` with router.replace, keeping the other parameters.
 *
 * Tab bodies are stubbed; the tab strip is a stand-in that shows the open tab.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen } from '@testing-library/react'

const { nav, role } = vi.hoisted(() => ({
  nav: { search: '', replace: vi.fn() },
  role: { isAdmin: true },
}))

vi.mock('next/link', () => ({
  default: ({ href, children }: any) => <a href={href}>{children}</a>,
}))
vi.mock('next/navigation', () => ({
  useRouter: () => ({ replace: nav.replace, push: vi.fn() }),
  useSearchParams: () => new URLSearchParams(nav.search),
  usePathname: () => '/settings',
}))
vi.mock('@/contexts/role-context', () => ({
  useSystemRole: () => ({ isAdmin: role.isAdmin, systemRole: role.isAdmin ? 'super_admin' : 'user', isLoading: false }),
}))
vi.mock('@/components/shared', () => ({
  PageHeader: () => null,
  FilterTabs: ({ tabs, value, onValueChange, children }: any) => (
    <div>
      <output data-testid="active-tab">{value}</output>
      {tabs.map((tab: any) => (
        <button key={tab.value} type="button" onClick={() => onValueChange(tab.value)}>
          {tab.label}
        </button>
      ))}
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
vi.mock('../SessionModeTab', () => ({ SessionModeTab: () => null }))
vi.mock('../WorkspaceSocialsCard', () => ({ WorkspaceSocialsCard: () => null }))
vi.mock('../WorkspaceVoiceLiveCard', () => ({ WorkspaceVoiceLiveCard: () => null }))
vi.mock('../themes/appearance-tab', () => ({ AppearanceTab: () => null }))
vi.mock('../phones-devices/phones-devices-tab', () => ({ PhonesDevicesTab: () => null }))
vi.mock('../billing/billing-tab', () => ({ BillingTab: () => null }))
vi.mock('../what-auto-depends-on', () => ({ WhatAutoDependsOn: () => null }))  // PRD-268 US-W05

async function loadPanel(edition: 'local' | 'saas') {
  vi.doMock('@/lib/auth-edition', () => ({
    authEdition: edition,
    isLocal: edition === 'local',
    isSaaS: edition === 'saas',
  }))
  const { SettingsPanel } = await import('../SettingsPanel')
  return SettingsPanel
}

const activeTab = () => screen.getByTestId('active-tab').textContent

beforeEach(() => {
  nav.search = ''
  nav.replace.mockReset()
  role.isAdmin = true
})

afterEach(() => {
  vi.doUnmock('@/lib/auth-edition')
  vi.resetModules()
})

describe('Settings ?tab= (PRD-265 US-011)', () => {
  it('opens the Billing tab from /settings?tab=billing', async () => {
    nav.search = 'tab=billing'
    const SettingsPanel = await loadPanel('saas')
    render(<SettingsPanel />)

    expect(activeTab()).toBe('billing')
  })

  it("lands on Billing from Auto's onboarding link, which also names a plan", async () => {
    nav.search = 'tab=billing&plan=pro'
    const SettingsPanel = await loadPanel('saas')
    render(<SettingsPanel />)

    expect(activeTab()).toBe('billing')
  })

  it('keeps the default tab when the address names none, or one that does not exist', async () => {
    const SettingsPanel = await loadPanel('saas')
    const { unmount } = render(<SettingsPanel />)
    expect(activeTab()).toBe('system-settings')
    unmount()

    nav.search = 'tab=nope'
    role.isAdmin = false
    render(<SettingsPanel />)
    expect(activeTab()).toBe('orchestrator')
  })

  it('a non-admin asked for System Settings gets the default instead', async () => {
    nav.search = 'tab=system-settings'
    role.isAdmin = false
    const SettingsPanel = await loadPanel('saas')
    render(<SettingsPanel />)

    expect(activeTab()).toBe('orchestrator')
  })

  it('the local edition has no Billing tab, so ?tab=billing opens the default', async () => {
    nav.search = 'tab=billing'
    const SettingsPanel = await loadPanel('local')
    render(<SettingsPanel />)

    expect(screen.queryByRole('button', { name: 'Billing' })).toBeNull()
    expect(activeTab()).toBe('system-settings')
  })

  it('choosing a tab switches at once and rewrites ?tab= with router.replace, keeping other parameters', async () => {
    nav.search = 'tab=billing&plan=pro'
    const SettingsPanel = await loadPanel('saas')
    render(<SettingsPanel />)

    fireEvent.click(screen.getByRole('button', { name: 'Webhooks' }))

    expect(activeTab()).toBe('webhooks')
    expect(nav.replace).toHaveBeenCalledTimes(1)
    expect(nav.replace).toHaveBeenCalledWith('/settings?tab=webhooks&plan=pro', { scroll: false })
  })
})
