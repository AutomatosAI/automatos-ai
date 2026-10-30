/**
 * PRD-261 A-07d — Settings › Appearance › Themes: who sees what, the labels, the opt-in, the choice.
 *
 * - Role gating: an owner or admin (workspace:manage) gets Install theme, the manage actions and
 *   the workspace's theme settings; every other member sees the list and the choice only.
 * - Labels: "Private · not reviewed" on unreviewed code, "Needs update" on a clone whose core
 *   page changed (the A-07b drift flag), "Update available" with the pending version.
 * - The opt-in dialog shows the server's wording and posts the packageId it showed; once the
 *   server recorded it, the choice is stored and the style context reloads into the theme.
 *
 * apiClient is mocked; the step-up path through the real apiClient is in
 * lib/theme-host/__tests__/api-step-up.test.tsx.
 */
import { readFileSync } from 'fs'
import path from 'path'

import { fireEvent, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const api = vi.hoisted(() => ({
  getInstalledThemes: vi.fn(),
  getThemeSettings: vi.fn(),
  getActiveTheme: vi.fn(),
  putThemeChoice: vi.fn(),
  importTheme: vi.fn(),
  optInTheme: vi.fn(),
  setDefaultTheme: vi.fn(),
  compareThemeUpdate: vi.fn(),
  updateThemeInstall: vi.fn(),
  rollbackTheme: vi.fn(),
  uninstallTheme: vi.fn(),
  putThemeSettings: vi.fn(),
}))
const workspaceState = vi.hoisted(() => ({ role: 'owner' }))
const setStyle = vi.hoisted(() => vi.fn())

vi.mock('@/lib/api-client', () => ({ apiClient: api }))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspace: () => ({ workspace: { id: 'ws-1', name: 'Acme', role: workspaceState.role } }),
}))
vi.mock('@/contexts/ui-style-context', () => ({ useUiStyleOptional: () => ({ setStyle }) }))
vi.mock('@clerk/nextjs', () => ({ useReverification: (fetcher: unknown) => fetcher }))
vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }))

import { AppearanceTab } from '../appearance-tab'
import { OPT_IN_CONFIRM } from '../opt-in-dialog'
import { USE_THEME } from '../installed-row'
import {
  MEMBERS_SEE_ONLY,
  NEEDS_UPDATE,
  PRIVATE_NOT_REVIEWED,
  UNREVIEWED_NOT_DEFAULT,
  UPDATES_ITSELF_HINT,
} from '../theme-labels'
import { INSTALL_ID, OPT_IN_WORDING, PACKAGE_ID, installRow, renderWithQuery, themeSettings } from './theme-fixtures'
import type { InstalledTheme } from '@/lib/api-client'

function serve(installs: InstalledTheme[], choice: string | null = 'studio') {
  api.getInstalledThemes.mockResolvedValue({ installs })
  api.getActiveTheme.mockResolvedValue({ choice, theme: null })
}

async function renderTab(role: string) {
  workspaceState.role = role
  renderWithQuery(<AppearanceTab />)
  return screen.findByTestId(`installed-theme-${INSTALL_ID}`)
}

beforeEach(() => {
  for (const mock of Object.values(api)) mock.mockReset()
  setStyle.mockReset()
  api.getThemeSettings.mockResolvedValue(themeSettings())
  api.putThemeChoice.mockResolvedValue({ choice: INSTALL_ID, theme: null })
  api.optInTheme.mockResolvedValue({ installId: INSTALL_ID, optIn: { required: false, wording: OPT_IN_WORDING, packageId: PACKAGE_ID } })
  serve([installRow()])
})

afterEach(() => {
  vi.clearAllMocks()
})

describe('role gating', () => {
  it.each(['owner', 'admin'])('%s: Install theme, the manage actions and the theme settings', async (role) => {
    const row = await renderTab(role)

    expect(screen.getByRole('button', { name: 'Install theme' })).toBeInTheDocument()
    expect(within(row).getByTestId('theme-manage-actions')).toBeInTheDocument()
    expect(within(row).getByRole('button', { name: 'Uninstall' })).toBeInTheDocument()
    expect(await screen.findByTestId('theme-settings')).toBeInTheDocument()
    expect(screen.queryByText(MEMBERS_SEE_ONLY)).not.toBeInTheDocument()
  })

  it.each(['editor', 'viewer', 'member'])('%s: the list and the choice only', async (role) => {
    const row = await renderTab(role)

    expect(within(row).getByRole('button', { name: USE_THEME })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Install theme' })).not.toBeInTheDocument()
    expect(screen.queryByTestId('theme-manage-actions')).not.toBeInTheDocument()
    expect(screen.queryByTestId('theme-settings')).not.toBeInTheDocument()
    expect(screen.getByText(MEMBERS_SEE_ONLY)).toBeInTheDocument()
  })

  it('an unreviewed theme cannot be made the workspace default, and the row says why', async () => {
    const row = await renderTab('owner')
    expect(within(row).getByRole('button', { name: 'Make default' })).toBeDisabled()
    expect(within(row).getByText(UNREVIEWED_NOT_DEFAULT)).toBeVisible()
  })
})

describe('labels', () => {
  it('shows "Private · not reviewed" on unreviewed code only', async () => {
    serve([installRow(), installRow({ installId: 'reviewed-1', themeId: 'acme/reviewed', reviewed: true, private: false })])
    const row = await renderTab('member')

    expect(within(row).getByText(PRIVATE_NOT_REVIEWED)).toBeInTheDocument()
    expect(within(screen.getByTestId('installed-theme-reviewed-1')).queryByText(PRIVATE_NOT_REVIEWED)).toBeNull()
  })

  it('shows "Needs update" for a clone whose core page changed, and not otherwise', async () => {
    serve([installRow({ needsUpdate: true }), installRow({ installId: 'fresh-1', needsUpdate: false })])
    const row = await renderTab('member')

    expect(within(row).getByTestId('theme-needs-update')).toHaveTextContent(NEEDS_UPDATE)
    expect(within(screen.getByTestId('installed-theme-fresh-1')).queryByTestId('theme-needs-update')).toBeNull()
  })

  it('shows "Update available" with the pending version', async () => {
    serve([installRow({ pendingPackageId: 'next', pendingVersion: '1.1.0' })])
    const row = await renderTab('member')
    expect(within(row).getByText('Update available · 1.1.0')).toBeInTheDocument()
  })

  it('says "Updates itself; pin to hold" on an unpinned Marketplace install only (publish contract D1)', async () => {
    const listed = { ownerType: 'marketplace', reviewed: true, private: false } as const
    serve([
      installRow(listed),
      installRow({ ...listed, installId: 'pinned-1', themeId: 'acme/pinned', pinnedVersion: '1.0.0' }),
      installRow({ installId: 'upload-1', themeId: 'acme/upload' }),
    ])
    const row = await renderTab('member')

    const badge = within(row).getByTestId('theme-updates-itself')
    expect(badge).toHaveTextContent('Updates itself; pin to hold')
    expect(badge).toHaveAttribute('title', UPDATES_ITSELF_HINT)
    const pinned = screen.getByTestId('installed-theme-pinned-1')
    expect(within(pinned).queryByTestId('theme-updates-itself')).toBeNull()
    expect(within(pinned).getByText('Pinned to 1.0.0')).toBeInTheDocument()
    expect(within(screen.getByTestId('installed-theme-upload-1')).queryByTestId('theme-updates-itself')).toBeNull()
  })

  it('marks the theme in use and offers no second choice of it', async () => {
    serve([installRow({ optIn: { required: false, wording: OPT_IN_WORDING, packageId: PACKAGE_ID } })], INSTALL_ID)
    const row = await renderTab('member')

    expect(await within(row).findByText('In use')).toBeInTheDocument()
    expect(within(row).queryByRole('button', { name: USE_THEME })).toBeNull()
  })
})

const OPT_IN_PY = path.resolve(__dirname, '..', '..', '..', '..', '..', 'orchestrator', 'services', 'themes', 'optin.py')

describe('the opt-in and the choice', () => {
  it("the fixture's wording is the server's own (services/themes/optin.py OPT_IN_WORDING)", () => {
    expect(readFileSync(OPT_IN_PY, 'utf8')).toContain(`OPT_IN_WORDING = "${OPT_IN_WORDING}"`)
  })

  it("shows the server's wording, posts the packageId it showed, then stores the choice and reloads into it", async () => {
    const row = await renderTab('member')
    fireEvent.click(within(row).getByRole('button', { name: USE_THEME }))

    const dialog = await screen.findByTestId('theme-opt-in-dialog')
    expect(within(dialog).getByTestId('theme-opt-in-wording')).toHaveTextContent(
      'runs with your access to all your workspaces',
    )
    expect(api.putThemeChoice).not.toHaveBeenCalled()
    fireEvent.click(within(dialog).getByRole('button', { name: OPT_IN_CONFIRM }))

    await waitFor(() => expect(api.optInTheme).toHaveBeenCalledWith(INSTALL_ID, PACKAGE_ID))
    await waitFor(() => expect(api.putThemeChoice).toHaveBeenCalledWith(INSTALL_ID))
    await waitFor(() => expect(setStyle).toHaveBeenCalledWith(`theme:install:${INSTALL_ID}`))
  })

  it('chooses straight away when this member already opted in to this version', async () => {
    serve([installRow({ optIn: { required: false, wording: OPT_IN_WORDING, packageId: PACKAGE_ID } })])
    const row = await renderTab('member')
    fireEvent.click(within(row).getByRole('button', { name: USE_THEME }))

    await waitFor(() => expect(api.putThemeChoice).toHaveBeenCalledWith(INSTALL_ID))
    expect(api.optInTheme).not.toHaveBeenCalled()
    expect(screen.queryByTestId('theme-opt-in-dialog')).toBeNull()
  })

  it('a 409 (the install moved on) chooses nothing, refetches the list and closes the dialog', async () => {
    api.optInTheme.mockRejectedValue(Object.assign(new Error('superseded'), { status: 409, body: { detail: 'superseded' } }))
    const row = await renderTab('member')
    fireEvent.click(within(row).getByRole('button', { name: USE_THEME }))
    fireEvent.click(within(await screen.findByTestId('theme-opt-in-dialog')).getByRole('button', { name: OPT_IN_CONFIRM }))

    await waitFor(() => expect(api.getInstalledThemes).toHaveBeenCalledTimes(2))
    await waitFor(() => expect(screen.queryByTestId('theme-opt-in-dialog')).toBeNull())
    expect(api.putThemeChoice).not.toHaveBeenCalled()
    expect(setStyle).not.toHaveBeenCalled()
  })

  it('reopening after a 409 opts in to the refetched version, not the stale one', async () => {
    const NEW_PACKAGE = 'pkg-moved-on'
    api.optInTheme.mockRejectedValueOnce(Object.assign(new Error('superseded'), { status: 409, body: { detail: 'superseded' } }))
    const row = await renderTab('member')
    api.getInstalledThemes.mockResolvedValue({
      installs: [installRow({ optIn: { required: true, wording: OPT_IN_WORDING, packageId: NEW_PACKAGE } })],
    })
    fireEvent.click(within(row).getByRole('button', { name: USE_THEME }))
    fireEvent.click(within(await screen.findByTestId('theme-opt-in-dialog')).getByRole('button', { name: OPT_IN_CONFIRM }))
    await waitFor(() => expect(screen.queryByTestId('theme-opt-in-dialog')).toBeNull())
    await waitFor(() => expect(api.getInstalledThemes).toHaveBeenCalledTimes(2))

    const fresh = await screen.findByTestId(`installed-theme-${INSTALL_ID}`)
    fireEvent.click(within(fresh).getByRole('button', { name: USE_THEME }))
    fireEvent.click(within(await screen.findByTestId('theme-opt-in-dialog')).getByRole('button', { name: OPT_IN_CONFIRM }))

    await waitFor(() => expect(api.optInTheme).toHaveBeenLastCalledWith(INSTALL_ID, NEW_PACKAGE))
  })

  it('a revoked (disabled) theme cannot be chosen', async () => {
    serve([installRow({ revoked: { level: 'disable', reason: 'Withdrawn by its publisher.' } })])
    const row = await renderTab('member')

    expect(within(row).getByRole('button', { name: USE_THEME })).toBeDisabled()
    expect(within(row).getByText('Withdrawn by its publisher.')).toBeInTheDocument()
  })
})
