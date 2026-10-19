/**
 * PRD-264 EUAI-10 — /verify, the public check for an Automatos AI mark.
 *
 * - Picking a file fingerprints it on the device and sends only the SHA-256 (through
 *   apiClient) to GET /api/public/provenance/{sha256}; the answer reads yes, with the
 *   kind and the date, or no. Too many checks read as a rate limit, not a failure.
 * - The page links contentcredentials.org/verify for C2PA files.
 * - It needs no sign-in in either edition: the saas middleware lists /verify as public,
 *   the local edition serves it, and it is a host-owned route no theme can take.
 *
 * The page header is stubbed (its theme slots are not under test); the fingerprint is
 * stubbed so jsdom needs no Web Crypto (lib/__tests__/provenance.test.ts covers it).
 */
import { readFileSync } from 'fs'
import path from 'path'
import React from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'

const HASH = 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'
const get = vi.hoisted(() => vi.fn())
const fingerprint = vi.hoisted(() => vi.fn())

vi.mock('@/lib/api-client', () => ({ apiClient: { get } }))
vi.mock('@/lib/provenance', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@/lib/provenance')>()),
  sha256Hex: fingerprint,
}))
vi.mock('@/components/shared/page-header', () => ({
  PageHeader: ({ title, titleAccent }: { title: string; titleAccent: string }) => (
    <h1>
      {title} {titleAccent}
    </h1>
  ),
}))

import VerifyPage from '@/app/verify/page'
import { HOST_OWNED_ROUTES } from '@/lib/theme-host/package-manifest'

function pickFile(name = 'clip.mp4') {
  const input = screen.getByLabelText('Choose a file to check')
  fireEvent.change(input, { target: { files: [new File(['frames'], name, { type: 'video/mp4' })] } })
}

beforeEach(() => {
  fingerprint.mockResolvedValue(HASH)
})

afterEach(() => {
  cleanup()
  get.mockReset()
  fingerprint.mockReset()
  vi.unstubAllEnvs()
  vi.resetModules()
})

describe('/verify — check a file without uploading it', () => {
  it('sends only the fingerprint and shows a yes with the kind and the date', async () => {
    get.mockResolvedValueOnce({ made_by_automatos_ai: true, kind: 'video', date: '2026-10-19' })
    render(<VerifyPage />)

    pickFile()

    expect(await screen.findByText('Made by Automatos AI: yes')).toBeInTheDocument()
    expect(screen.getByText(/Video · made on 2026-10-19/)).toBeInTheDocument()
    expect(screen.getByTestId('provenance-hash')).toHaveTextContent(HASH)
    expect(get).toHaveBeenCalledTimes(1)
    expect(get.mock.calls[0]).toEqual([`/api/public/provenance/${HASH}`])
  })

  it('shows a no, and says an edited copy reads no', async () => {
    get.mockResolvedValueOnce({ made_by_automatos_ai: false })
    render(<VerifyPage />)

    pickFile('photo.png')

    expect(await screen.findByText('Made by Automatos AI: no')).toBeInTheDocument()
    expect(screen.getByTestId('provenance-answer')).toHaveTextContent(/re-encoded has a\s+different fingerprint/)
  })

  it('says so when the address has used its checks for the minute', async () => {
    get.mockRejectedValueOnce(Object.assign(new Error('Too many checks'), { status: 429 }))
    render(<VerifyPage />)

    pickFile()

    expect(await screen.findByRole('alert')).toHaveTextContent('Too many checks from your address')
  })

  it('links Content Credentials for C2PA files', () => {
    render(<VerifyPage />)
    const link = screen.getByRole('link', { name: 'contentcredentials.org/verify' })
    expect(link).toHaveAttribute('href', 'https://contentcredentials.org/verify')
    expect(link).toHaveAttribute('rel', 'noopener noreferrer')
  })
})

describe('/verify — public in both editions, owned by the host', () => {
  const middleware = readFileSync(path.resolve(__dirname, '..', '..', 'middleware.ts'), 'utf8')

  it('is a public route of the saas middleware', () => {
    const publicList = middleware.slice(middleware.indexOf('createRouteMatcher(['), middleware.indexOf(']);'))
    expect(publicList).toContain('"/verify(.*)"')
  })

  it('exists in the local edition', async () => {
    vi.stubEnv('NEXT_PUBLIC_AUTH_EDITION', 'local')
    vi.resetModules()
    const edition = await import('@/lib/auth-edition')
    expect(edition.isLocal).toBe(true)
    expect(edition.isRouteAvailableInEdition('/verify')).toBe(true)
  })

  it('is a host-owned route, so no theme can replace it', () => {
    expect(HOST_OWNED_ROUTES).toContain('/verify')
  })
})
