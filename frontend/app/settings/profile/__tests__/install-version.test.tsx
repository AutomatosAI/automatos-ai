/**
 * PRD-279 US-303 — the profile page shows the version the API reports, beside the edition, in
 * every edition: GET /api/system/health's `version`, read through apiClient. Nothing is shown
 * when the read fails or the answer holds no version of the release's or a commit id's form.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'

const { getSystemHealthMock } = vi.hoisted(() => ({ getSystemHealthMock: vi.fn() }))

vi.mock('@/lib/api-client', () => ({ apiClient: { getSystemHealth: getSystemHealthMock } }))

import { EDITION_LABELS, InstallVersion, reportedVersion } from '../install-version'

afterEach(() => {
  getSystemHealthMock.mockReset()
  vi.restoreAllMocks()
})

describe('the version beside the edition (PRD-279 US-303)', () => {
  it.each([
    ['local', 'Local edition'],
    ['saas', 'Hosted edition'],
    ['enterprise', 'Enterprise edition'],
  ] as const)('%s shows its edition and the version the API reports', async (edition, label) => {
    getSystemHealthMock.mockResolvedValue({ overall_status: 'healthy', version: '0.12.0-rc.1' })
    render(<InstallVersion edition={edition} />)

    const line = await screen.findByTestId('install-version')
    expect(line).toHaveTextContent(`${label} · version 0.12.0-rc.1`)
    expect(EDITION_LABELS[edition]).toBe(label)
    expect(getSystemHealthMock).toHaveBeenCalledTimes(1)
  })

  it('shows a build from main by its commit id', async () => {
    const commit = '0d9f0ac594c3783e62a30e6c5ce75164028850db'
    getSystemHealthMock.mockResolvedValue({ version: commit })
    render(<InstallVersion edition="local" />)
    expect(await screen.findByTestId('install-version')).toHaveTextContent(`Local edition · version ${commit}`)
  })

  it('shows nothing, and logs, when the API does not answer', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    getSystemHealthMock.mockRejectedValue(new Error('The API is not reachable.'))
    render(<InstallVersion edition="enterprise" />)

    await waitFor(() => expect(warn).toHaveBeenCalled())
    expect(screen.queryByTestId('install-version')).toBeNull()
  })

  it('reads only a version of the release form or a commit id from the answer', () => {
    expect(reportedVersion({ version: '0.12.0' })).toBe('0.12.0')
    expect(reportedVersion({ version: 'dev' })).toBe('dev')
    for (const answer of [null, undefined, 'x', {}, { version: 12 }, { version: '' },
      { version: '<b>0.12.0</b>' }, { version: '0.12.0\nnext' }, { version: '-rc' }, { version: 'a'.repeat(129) }]) {
      expect(reportedVersion(answer)).toBeNull()
    }
  })
})
