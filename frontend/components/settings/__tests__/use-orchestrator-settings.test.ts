/**
 * PRD-263 US-017 — picking a HARNESS mode keeps the rest of the HARNESS settings.
 *
 * HarnessSettingsSection sends only ('mode', value). useOrchestratorConfig merges it
 * into the loaded HARNESS block, so the save still carries the schedule the API
 * returned, and the other settings blocks are untouched.
 */
import { describe, it, expect, vi } from 'vitest'
import { act, renderHook, waitFor } from '@testing-library/react'

const requestMock = vi.hoisted(() => vi.fn())

vi.mock('@/lib/api-client', () => ({
  apiClient: { request: requestMock },
}))

import { useOrchestratorConfig } from '../use-orchestrator-settings'
import { fallbackOrchestratorConfig } from '../orchestrator-settings-config'

describe('useOrchestratorConfig().handleHarnessChange (PRD-263 US-017)', () => {
  it('sets the mode and keeps the schedule and the other blocks', async () => {
    const loaded = { ...fallbackOrchestratorConfig(), harness: { schedule: 'weekly', mode: 'report_only' } }
    requestMock.mockResolvedValue(loaded)

    const { result } = renderHook(() => useOrchestratorConfig())
    await waitFor(() => expect(result.current.orchLoading).toBe(false))

    act(() => result.current.handleHarnessChange('mode', 'disabled'))

    expect(result.current.orchConfig?.harness).toEqual({ schedule: 'weekly', mode: 'disabled' })
    expect(result.current.orchConfig?.heartbeat).toEqual(loaded.heartbeat)
    expect(loaded.harness).toEqual({ schedule: 'weekly', mode: 'report_only' })
  })
})
