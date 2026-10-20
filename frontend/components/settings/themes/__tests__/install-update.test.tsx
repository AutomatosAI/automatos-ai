/**
 * PRD-261 A-07d — Install theme, the scan report, update (compare and apply), rollback, uninstall
 * and the workspace's theme settings, as an owner sees them.
 *
 * `@clerk/nextjs` is a double that does what Clerk's `createReverificationHandler` does
 * (`@clerk/shared` react/index.mjs:1529-1576) when the caller passes no `onNeedsReverification`:
 * call the fetcher; on Clerk's own `isReverificationHint(result)`, open Clerk's reverification
 * (here `openReverification`, which completes), and call the fetcher ONE more time. So a step-up
 * 403 from the upload or a settings change is proved to reach Clerk from the component.
 */
import { fireEvent, screen, waitFor, within } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

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
const clerk = vi.hoisted(() => ({ openReverification: vi.fn() }))

vi.mock('@/lib/api-client', () => ({ apiClient: api }))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspace: () => ({ workspace: { id: 'ws-1', name: 'Acme', role: 'owner' } }),
}))
vi.mock('@/contexts/ui-style-context', () => ({ useUiStyleOptional: () => ({ setStyle: vi.fn() }) }))
vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }))
vi.mock('@clerk/nextjs', async () => {
  const { isReverificationHint } = await import('@clerk/shared/authorization-errors')
  type Fetcher = (...args: unknown[]) => Promise<unknown>
  return {
    useReverification: (fetcher: Fetcher) => async (...args: unknown[]) => {
      let result = await fetcher(...args)
      if (isReverificationHint(result)) {
        await clerk.openReverification()
        result = await fetcher(...args)
      }
      return result
    },
  }
})

import { AppearanceTab } from '../appearance-tab'
import { APPLY_UPDATE } from '../update-compare'
import { INSTALL_THEME, UPLOADS_OFF } from '../install-theme'
import { PRIVATE_NOT_REVIEWED } from '../theme-labels'
import {
  INSTALL_ID,
  NEXT_PACKAGE_ID,
  PACKAGE_ID,
  installRow,
  renderWithQuery,
  stepUpError,
  themeSettings,
} from './theme-fixtures'

const SCAN_REPORT = {
  schema: 'automatos-theme-scan/v1',
  verdict: 'not_run',
  checks: {
    sizes: { status: 'pass', packageBytes: 20480, codeBytes: 10240, entryBytes: 2048 },
    cssScope: { status: 'pass', path: 'dist/theme.css' },
    looks: { status: 'pass', presets: ['default'] },
    contrast: { status: 'not_run', reason: 'Not measured: no colour pairs.', presets: [{ id: 'default', status: 'not_run' }] },
    automated: { status: 'not_run', reason: 'The automated review stage did not run.' },
  },
}

const UPLOAD = {
  package: {
    id: PACKAGE_ID, themeId: 'acme/hello', version: '1.0.0', name: 'Hello theme', digest: 'a'.repeat(64),
    ownerType: 'workspace', reviewed: false, scanVerdict: 'not_run', scanReport: SCAN_REPORT,
  },
  install: { id: INSTALL_ID, themeId: 'acme/hello', packageId: PACKAGE_ID, pendingPackageId: null, isDefault: false },
  status: 'installed',
  duplicate: false,
}

const COMPARISON = {
  installId: INSTALL_ID,
  from: { packageId: PACKAGE_ID, version: '1.0.0', reviewed: false },
  to: { packageId: NEXT_PACKAGE_ID, version: '1.1.0', reviewed: false },
  files: { added: ['dist/extra.js'], removed: [], changed: ['dist/entry.js'] },
  scan: { verdict: { from: 'not_run', to: 'not_run' }, changed: { checks: { from: 1, to: 2 } } },
}

function chooseFile(): File {
  const file = new File(['zip'], 'hello.autotheme', { type: 'application/zip' })
  fireEvent.change(screen.getByTestId('install-theme-input'), { target: { files: [file] } })
  return file
}

async function renderOwnerTab() {
  renderWithQuery(<AppearanceTab />)
  return screen.findByTestId(`installed-theme-${INSTALL_ID}`)
}

beforeEach(() => {
  for (const mock of Object.values(api)) mock.mockReset()
  clerk.openReverification.mockReset()
  clerk.openReverification.mockResolvedValue(undefined)
  api.getInstalledThemes.mockResolvedValue({ installs: [installRow()] })
  api.getActiveTheme.mockResolvedValue({ choice: 'studio', theme: null })
  api.getThemeSettings.mockResolvedValue(themeSettings())
})

describe('Install theme', () => {
  it('uploads the package and shows the scan report with "Private · not reviewed"', async () => {
    api.importTheme.mockResolvedValue(UPLOAD)
    await renderOwnerTab()
    const file = chooseFile()

    await waitFor(() => expect(api.importTheme).toHaveBeenCalledWith(file))
    const result = await screen.findByTestId('theme-upload-result')
    expect(within(result).getByText(PRIVATE_NOT_REVIEWED)).toBeInTheDocument()
    const report = within(result).getByTestId('theme-scan-report')
    expect(within(report).getByText('Sizes')).toBeInTheDocument()
    expect(within(report).getByText('Contrast (AA)')).toBeInTheDocument()
    expect(within(report).getByText('Not measured: no colour pairs.')).toBeInTheDocument()
    expect(within(report).getAllByText('Not run').length).toBeGreaterThanOrEqual(2)
    expect(clerk.openReverification).not.toHaveBeenCalled()
  })

  it('a step-up 403 opens Clerk once and the upload is sent again', async () => {
    api.importTheme.mockRejectedValueOnce(stepUpError()).mockResolvedValueOnce(UPLOAD)
    await renderOwnerTab()
    chooseFile()

    expect(await screen.findByTestId('theme-upload-result')).toBeInTheDocument()
    expect(clerk.openReverification).toHaveBeenCalledTimes(1)
    expect(api.importTheme).toHaveBeenCalledTimes(2)
  })

  it('is off, with the reason, when private uploads are off here', async () => {
    api.getThemeSettings.mockResolvedValue(
      themeSettings({ privateUploads: { setting: 'off', source: 'workspace', default: 'on', platform: 'auto', effective: 'off' } }),
    )
    await renderOwnerTab()

    expect(await screen.findByText(UPLOADS_OFF)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: INSTALL_THEME })).toBeDisabled()
  })

  it('is off for an admin when only the owner may install', async () => {
    api.getThemeSettings.mockResolvedValue(themeSettings({ canInstall: false, installRole: 'owner' }))
    await renderOwnerTab()

    expect(await screen.findByText(/only the workspace owner can install themes/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: INSTALL_THEME })).toBeDisabled()
  })
})

describe('update, rollback and uninstall', () => {
  it('compares the pending version and applies the package the compare showed', async () => {
    api.getInstalledThemes.mockResolvedValue({
      installs: [installRow({ pendingPackageId: NEXT_PACKAGE_ID, pendingVersion: '1.1.0' })],
    })
    api.compareThemeUpdate.mockResolvedValue(COMPARISON)
    api.updateThemeInstall.mockResolvedValue(installRow({ version: '1.1.0' }))
    const row = await renderOwnerTab()
    fireEvent.click(within(row).getByRole('button', { name: 'Review update' }))

    const compare = await screen.findByTestId('theme-update-compare')
    expect(api.compareThemeUpdate).toHaveBeenCalledWith(INSTALL_ID, NEXT_PACKAGE_ID)
    expect(within(compare).getByText('Version 1.0.0 → 1.1.0')).toBeInTheDocument()
    expect(within(compare).getByText('dist/entry.js')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: APPLY_UPDATE }))

    await waitFor(() => expect(api.updateThemeInstall).toHaveBeenCalledWith(INSTALL_ID, NEXT_PACKAGE_ID))
  })

  it('a 409 on Apply (the install already moved on) refetches the list and closes the compare', async () => {
    api.getInstalledThemes.mockResolvedValue({
      installs: [installRow({ pendingPackageId: NEXT_PACKAGE_ID, pendingVersion: '1.1.0' })],
    })
    api.compareThemeUpdate.mockResolvedValue(COMPARISON)
    api.updateThemeInstall.mockRejectedValue(Object.assign(new Error('moved on'), { status: 409, body: { detail: 'moved on' } }))
    const row = await renderOwnerTab()
    fireEvent.click(within(row).getByRole('button', { name: 'Review update' }))
    await screen.findByTestId('theme-update-compare')
    fireEvent.click(screen.getByRole('button', { name: APPLY_UPDATE }))

    await waitFor(() => expect(api.getInstalledThemes).toHaveBeenCalledTimes(2))
    await waitFor(() => expect(screen.queryByTestId('theme-update-compare')).toBeNull())
  })

  it('rolls back to the previous version', async () => {
    api.getInstalledThemes.mockResolvedValue({ installs: [installRow({ previousPackageId: 'previous-1' })] })
    api.rollbackTheme.mockResolvedValue(installRow())
    const row = await renderOwnerTab()
    fireEvent.click(within(row).getByRole('button', { name: 'Roll back' }))

    await waitFor(() => expect(api.rollbackTheme).toHaveBeenCalledWith(INSTALL_ID))
  })

  it('uninstalls only after a second, confirming click', async () => {
    api.uninstallTheme.mockResolvedValue({ uninstalled: INSTALL_ID })
    const row = await renderOwnerTab()
    fireEvent.click(within(row).getByRole('button', { name: 'Uninstall' }))
    expect(api.uninstallTheme).not.toHaveBeenCalled()
    fireEvent.click(within(row).getByRole('button', { name: 'Confirm uninstall' }))

    await waitFor(() => expect(api.uninstallTheme).toHaveBeenCalledWith(INSTALL_ID))
  })

  it('makes a reviewed theme the workspace default', async () => {
    api.getInstalledThemes.mockResolvedValue({ installs: [installRow({ reviewed: true, private: false })] })
    api.setDefaultTheme.mockResolvedValue({ defaultInstallId: INSTALL_ID })
    const row = await renderOwnerTab()
    fireEvent.click(within(row).getByRole('button', { name: 'Make default' }))

    await waitFor(() => expect(api.setDefaultTheme).toHaveBeenCalledWith(INSTALL_ID))
  })
})

describe('the workspace theme settings (A-07e)', () => {
  it('a step-up 403 on a change opens Clerk once and the change is sent again', async () => {
    api.putThemeSettings.mockRejectedValueOnce(stepUpError()).mockResolvedValueOnce(themeSettings({ membersMayChoose: false }))
    await renderOwnerTab()
    const settings = await screen.findByTestId('theme-settings')
    fireEvent.click(within(settings).getByRole('switch', { name: 'Members choose their own theme' }))

    await waitFor(() => expect(api.putThemeSettings).toHaveBeenCalledTimes(2))
    expect(api.putThemeSettings).toHaveBeenNthCalledWith(2, { membersMayChoose: false })
    expect(clerk.openReverification).toHaveBeenCalledTimes(1)
  })

  it('pins an install to one of its versions', async () => {
    api.getThemeSettings.mockResolvedValue(
      themeSettings({
        pins: [{ installId: INSTALL_ID, themeId: 'acme/hello', version: '1.0.0', pinnedVersion: null, pinnableVersions: ['1.0.0'] }],
      }),
    )
    api.putThemeSettings.mockResolvedValue(themeSettings())
    await renderOwnerTab()
    const pin = await screen.findByLabelText('acme/hello')
    fireEvent.change(pin, { target: { value: '1.0.0' } })

    await waitFor(() =>
      expect(api.putThemeSettings).toHaveBeenCalledWith({ pins: [{ installId: INSTALL_ID, version: '1.0.0' }] }),
    )
  })

  it('changes who may install', async () => {
    api.putThemeSettings.mockResolvedValue(themeSettings({ installRole: 'owner' }))
    await renderOwnerTab()
    fireEvent.change(await screen.findByLabelText('Who may install themes'), { target: { value: 'owner' } })

    await waitFor(() => expect(api.putThemeSettings).toHaveBeenCalledWith({ installRole: 'owner' }))
  })
})
