/**
 * PRD-277 US-704 (D-12) — the admin's theme revocations on an enterprise build, where pulling Automatos's
 * revocation list is the operator's own setting (GET /api/hosted-services).
 *
 * - switched off: the sync section says so in one plain line, the sync status is never asked, and a feed row
 *   (from a time it was on) has Lift, as the server's rule allows while the install doesn't pull;
 * - switched on: the sync status shows as on the local edition, with Check now;
 * - while the setting is unknown the sync status is not asked, no off line shows, and a feed row has no Lift;
 * - `pullsRevocations`: never hosted, always local, the setting in enterprise.
 *
 * `@clerk/nextjs` is the double revocations-panel.test.tsx uses (create and lift carry step-up).
 */
import { screen, waitFor, within } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const api = vi.hoisted(() => ({
  get: vi.fn(),
  getThemeRevocations: vi.fn(),
  createThemeRevocation: vi.fn(),
  liftThemeRevocation: vi.fn(),
  getThemeRevocationSync: vi.fn(),
  pullThemeRevocations: vi.fn(),
}))

vi.mock('@/lib/edition-name', () => ({ isEnterprise: true, resolveEdition: () => 'enterprise' }))
vi.mock('@/lib/api-client', () => ({ apiClient: api }))
vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }))
vi.mock('@clerk/nextjs', () => ({
  useReverification: (fetcher: (...args: unknown[]) => Promise<unknown>) => fetcher,
}))

import { HOSTED_SERVICES_ENDPOINT } from '@/lib/hosted-services'

import { RevocationsPanel, pullsRevocations } from '../revocations-panel'
import { CHECK_NOW, LIFT, SYNC_NOT_SWITCHED_ON } from '../review-words'
import { FEED_REVOCATION_ID, renderWithQuery, revocationRow, syncStatus } from './review-fixtures'

const FEED_ROW = revocationRow({ id: FEED_REVOCATION_ID, origin: 'feed', digest: null, themeId: 'acme/ink', versionRange: '^1.0.0', level: 'warn' })

function serve(themeRevocations: boolean | 'pending') {
  api.get.mockImplementation((path: string) => {
    if (path !== HOSTED_SERVICES_ENDPOINT) throw new Error(`unexpected request ${path}`)
    if (themeRevocations === 'pending') return new Promise(() => {})
    return Promise.resolve({ themeRevocations, cloudSync: false, cloudItems: false })
  })
  api.getThemeRevocations.mockResolvedValue({ revocations: [FEED_ROW] })
  api.getThemeRevocationSync.mockResolvedValue(syncStatus())
}

const feedRow = () => screen.findByTestId(`theme-revocation-${FEED_REVOCATION_ID}`)

beforeEach(() => {
  for (const mock of Object.values(api)) mock.mockReset()
})

describe('the revocation sync on an enterprise build', () => {
  it('switched off: one plain line, no status asked, and a feed row can be lifted', async () => {
    serve(false)
    renderWithQuery(<RevocationsPanel hosted={false} />)

    expect(await screen.findByTestId('revocation-sync-not-switched-on')).toHaveTextContent(SYNC_NOT_SWITCHED_ON)
    expect(await within(await feedRow()).findByRole('button', { name: LIFT })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: CHECK_NOW })).not.toBeInTheDocument()
    expect(api.getThemeRevocationSync).not.toHaveBeenCalled()
  })

  it('switched on: the sync status and Check now, as on the local edition', async () => {
    serve(true)
    renderWithQuery(<RevocationsPanel hosted={false} />)

    expect(await screen.findByTestId('revocation-sync-success')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: CHECK_NOW })).toBeEnabled()
    expect(screen.queryByTestId('revocation-sync-not-switched-on')).not.toBeInTheDocument()
    expect(within(await feedRow()).queryByRole('button', { name: LIFT })).toBeNull()
  })

  it('while the setting is unknown: no status asked, no off line, no Lift on a feed row', async () => {
    serve('pending')
    renderWithQuery(<RevocationsPanel hosted={false} />)

    const row = await feedRow()
    await waitFor(() => expect(api.get).toHaveBeenCalledWith(HOSTED_SERVICES_ENDPOINT))
    expect(within(row).queryByRole('button', { name: LIFT })).toBeNull()
    expect(screen.queryByTestId('revocation-sync-not-switched-on')).not.toBeInTheDocument()
    expect(api.getThemeRevocationSync).not.toHaveBeenCalled()
  })
})

describe('pullsRevocations', () => {
  it('never on the hosted edition, always locally, the setting in enterprise', () => {
    expect(pullsRevocations(true, false, undefined)).toBe(false)
    expect(pullsRevocations(true, true, true)).toBe(false)
    expect(pullsRevocations(false, false, undefined)).toBe(true)
    expect(pullsRevocations(false, true, true)).toBe(true)
    expect(pullsRevocations(false, true, false)).toBe(false)
    expect(pullsRevocations(false, true, undefined)).toBeUndefined()
  })
})
