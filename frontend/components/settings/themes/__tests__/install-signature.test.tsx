/**
 * PRD-261 C-05k — Install theme with the package's signature file, and "Signed by Automatos":
 *
 * - an optional `.sig.json` added before the `.autotheme` goes in the same upload (importTheme's
 *   second argument, the `signature` part); without one the package goes alone, as before;
 * - a refusal of the signature (400 `SIGNATURE_MISMATCH`) shows the server's own sentence;
 * - an installed row whose signature an active key verifies shows "Signed by Automatos" in place
 *   of "Private · not reviewed" (QB-24), `invalid` never does, and Make default follows the
 *   server's `private`.
 */
import { fireEvent, screen, waitFor, within } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const api = vi.hoisted(() => ({
  importTheme: vi.fn(),
  setDefaultTheme: vi.fn(),
  rollbackTheme: vi.fn(),
  uninstallTheme: vi.fn(),
  // renderWithQuery mounts the app's ChooseInstallProvider, whose opt-in dialog binds these when it mounts.
  optInTheme: vi.fn(),
  putThemeChoice: vi.fn(),
}))

vi.mock('@/lib/api-client', () => ({ apiClient: api }))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspace: () => ({ workspace: { id: 'ws-1', name: 'Acme', role: 'owner' } }),
}))
vi.mock('@/contexts/ui-style-context', () => ({ useUiStyleOptional: () => ({ setStyle: vi.fn() }) }))
vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }))
vi.mock('@clerk/nextjs', () => ({
  useReverification: (fetcher: (...args: unknown[]) => Promise<unknown>) => fetcher,
}))

import type { InstalledTheme, ThemeSignatureVerdict } from '@/lib/api-client'

import { InstallTheme } from '../install-theme'
import { InstalledRow } from '../installed-row'
import {
  PRIVATE_NOT_REVIEWED,
  SIGNATURE_WORDS,
  SIGNED_BY_AUTOMATOS,
  UNREVIEWED_NOT_DEFAULT,
  isSignedByAutomatos,
  signatureWords,
} from '../theme-labels'
import { INSTALL_ID, PACKAGE_ID, installRow, renderWithQuery, themeSettings } from './theme-fixtures'

const KEY_ID = 'ed25519:0123456789abcdef'
const VERIFIED: ThemeSignatureVerdict = { state: 'verified', keyId: KEY_ID }
const INVALID: ThemeSignatureVerdict = { state: 'invalid', keyId: null }
// api/themes_import.py IMPORT_REFUSED and signed_upload.py FOR_ANOTHER_PACKAGE (the theme id).
const IMPORT_REFUSED = "This theme can't be installed."
const FOR_ANOTHER_PACKAGE =
  "This signature is for another package: its theme id isn't the uploaded theme's. Upload the .sig.json made for this .autotheme."

const UPLOAD = {
  package: {
    id: PACKAGE_ID, themeId: 'acme/hello', version: '1.0.0', name: 'Hello theme', digest: 'a'.repeat(64),
    ownerType: 'workspace', reviewed: false, scanVerdict: 'review_required', scanReport: null,
  },
  install: { id: INSTALL_ID, themeId: 'acme/hello', packageId: PACKAGE_ID, pendingPackageId: null, isDefault: false },
  status: 'installed',
  duplicate: false,
}

function chooseSignature(): File {
  const file = new File(['{"form":"automatos-theme-signature/v1"}'], 'hello-1.0.0.sig.json', { type: 'application/json' })
  fireEvent.change(screen.getByTestId('install-theme-signature-input'), { target: { files: [file] } })
  return file
}

function choosePackage(): File {
  const file = new File(['zip'], 'hello.autotheme', { type: 'application/zip' })
  fireEvent.change(screen.getByTestId('install-theme-input'), { target: { files: [file] } })
  return file
}

function signatureRefusal(): Error {
  const detail = { message: IMPORT_REFUSED, errors: [{ code: 'SIGNATURE_MISMATCH', path: 'signature', message: FOR_ANOTHER_PACKAGE }] }
  return Object.assign(new Error(JSON.stringify(detail)), { status: 400, body: { detail } })
}

function renderRow(install: InstalledTheme) {
  renderWithQuery(
    <ul>
      <InstalledRow install={install} canManage inUse={false} choosing={false} onUse={vi.fn()} onReviewUpdate={vi.fn()} />
    </ul>,
  )
  return screen.getByTestId(`installed-theme-${install.installId}`)
}

beforeEach(() => {
  for (const mock of Object.values(api)) mock.mockReset()
})

describe('Install theme with a signature file (C-05k)', () => {
  it('sends the .sig.json beside the .autotheme, and the upload reads "Signed by Automatos"', async () => {
    api.importTheme.mockResolvedValue(UPLOAD)
    renderWithQuery(<InstallTheme settings={themeSettings()} />)
    const signature = chooseSignature()
    expect(screen.getByTestId('install-theme-signature-name')).toHaveTextContent('hello-1.0.0.sig.json')
    const file = choosePackage()

    await waitFor(() => expect(api.importTheme).toHaveBeenCalledWith(file, signature))
    const result = await screen.findByTestId('theme-upload-result')
    expect(within(result).getByText(SIGNED_BY_AUTOMATOS)).toBeInTheDocument()
    expect(within(result).queryByText(PRIVATE_NOT_REVIEWED)).toBeNull()
    expect(screen.queryByTestId('install-theme-signature-name')).toBeNull()
  })

  it('without a signature file sends the package alone, as before', async () => {
    api.importTheme.mockResolvedValue(UPLOAD)
    renderWithQuery(<InstallTheme settings={themeSettings()} />)
    const file = choosePackage()

    await waitFor(() => expect(api.importTheme).toHaveBeenCalledTimes(1))
    expect(api.importTheme.mock.calls[0]).toEqual([file])
    expect(within(await screen.findByTestId('theme-upload-result')).getByText(PRIVATE_NOT_REVIEWED)).toBeInTheDocument()
  })

  it('Remove drops the chosen signature file', async () => {
    api.importTheme.mockResolvedValue(UPLOAD)
    renderWithQuery(<InstallTheme settings={themeSettings()} />)
    chooseSignature()
    fireEvent.click(screen.getByRole('button', { name: 'Remove' }))
    expect(screen.queryByTestId('install-theme-signature-name')).toBeNull()
    const file = choosePackage()
    await waitFor(() => expect(api.importTheme.mock.calls[0]).toEqual([file]))
  })

  it("a 400 SIGNATURE_MISMATCH shows the server's sentence, and nothing is shown as installed", async () => {
    api.importTheme.mockRejectedValue(signatureRefusal())
    renderWithQuery(<InstallTheme settings={themeSettings()} />)
    chooseSignature()
    choosePackage()

    const refusal = await screen.findByTestId('theme-upload-refusal')
    expect(refusal).toHaveTextContent(FOR_ANOTHER_PACKAGE)
    expect(refusal).toHaveTextContent('SIGNATURE_MISMATCH')
    expect(screen.queryByTestId('theme-upload-result')).toBeNull()
    expect(screen.getByTestId('install-theme-signature-name')).toHaveTextContent('hello-1.0.0.sig.json')
  })
})

describe('"Signed by Automatos" on an installed row (QB-24)', () => {
  it('a verified signature shows it in place of "Private · not reviewed"', () => {
    const row = renderRow(installRow({ signature: VERIFIED, private: false }))
    expect(within(row).getByTestId('theme-signed-by-automatos')).toHaveTextContent(SIGNED_BY_AUTOMATOS)
    expect(within(row).queryByText(PRIVATE_NOT_REVIEWED)).toBeNull()
  })

  it.each([
    ['invalid', INVALID],
    ['unsigned', { state: 'unsigned', keyId: null } as ThemeSignatureVerdict],
  ])('an %s signature never reads as signed: the row stays "Private · not reviewed"', (_case, signature) => {
    const row = renderRow(installRow({ signature }))
    expect(within(row).queryByText(SIGNED_BY_AUTOMATOS)).toBeNull()
    expect(within(row).getByText(PRIVATE_NOT_REVIEWED)).toBeInTheDocument()
  })

  it("Make default follows the server's private: a signed upload may be the default, private code may not", () => {
    const signed = renderRow(installRow({ signature: VERIFIED, private: false }))
    expect(within(signed).getByRole('button', { name: 'Make default' })).toBeEnabled()
    expect(within(signed).queryByTestId('theme-default-reason')).toBeNull()
  })

  it('private code keeps Make default off, with the reason written beside it and as its title', () => {
    const row = renderRow(installRow({ signature: INVALID, private: true }))
    const makeDefault = within(row).getByRole('button', { name: 'Make default' })
    expect(makeDefault).toBeDisabled()
    expect(makeDefault).toHaveAttribute('title', UNREVIEWED_NOT_DEFAULT)
    const reason = within(row).getByTestId('theme-default-reason')
    expect(reason).toHaveTextContent(UNREVIEWED_NOT_DEFAULT)
    expect(reason).toBeVisible()
    expect(reason.previousElementSibling).toBe(makeDefault)
  })

  it('the workspace default shows Clear default and no reason', () => {
    const row = renderRow(installRow({ private: false, isDefault: true }))
    expect(within(row).getByRole('button', { name: 'Clear default' })).toBeEnabled()
    expect(within(row).queryByTestId('theme-default-reason')).toBeNull()
  })

  it('the words: only verified is "Signed by Automatos"; a missing verdict reads as not signed', () => {
    expect(signatureWords(VERIFIED)).toBe(SIGNED_BY_AUTOMATOS)
    expect(signatureWords(INVALID)).toBe(SIGNATURE_WORDS.invalid)
    expect(signatureWords(INVALID)).not.toContain(SIGNED_BY_AUTOMATOS)
    expect(signatureWords(undefined)).toBe(SIGNATURE_WORDS.unsigned)
    expect(isSignedByAutomatos({ signature: INVALID })).toBe(false)
    expect(isSignedByAutomatos({ signature: VERIFIED })).toBe(true)
  })
})
