/**
 * PRD-261 A-09b — the LookProvider as the app mounts it, on the real server client (apiClient's
 * named theme methods, mocked here): under Play, with a workspace, it reads builtin Play's looks
 * from GET /api/themes/looks/automatos/play and writes a change through PUT; a save the server
 * refuses brings back the Looks page's not-kept notice (M98), a save that just didn't reach it
 * does not (this browser keeps it and sends it on the next load).
 */
import { act, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { LookNotice, LOOKS_NOT_KEPT, storedNotice } from '@/components/looks/look-notice'
import { LookProvider, useLook } from '@/lib/theme-host/builtin/play-look-provider'
import { LookScopeContext } from '@/components/theme-host/look-scope'
import { LOOK_SEND_DELAY_MS } from '@/lib/theme-host/builtin/play-look-store'
import { UiStyleProvider } from '@/contexts/ui-style-context'
import { apiClient } from '@/lib/api-client'
import { clearRoot, memoryStorage } from '@/lib/looks/__tests__/helpers'
import { AURORA, DAYLIGHT } from '@/lib/looks/presets'
import { lookScope } from '@/lib/looks/scope'
import { hasLookMark, lookKeys, setLookMark } from '@/lib/looks/storage'
import { BUILTIN_PLAY_STYLE } from '@/lib/theme-host/builtin/play-entry'
import { BUILTIN_PLAY_LOOK_KEYS } from '@/lib/theme-host/builtin/play-look-keys'
import { APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION } from '@/lib/ui-style'

vi.mock('@/lib/api-client', () => {
  const client = { getThemeLooks: vi.fn(), putThemeLooks: vi.fn() }
  return { apiClient: client, default: client }
})

const SCOPE = lookScope('7d3c2a10-1b2c-4d5e-8f90-123456789abc', 'local')
const KEYS = lookKeys(SCOPE, BUILTIN_PLAY_LOOK_KEYS)
const SAVED_TEXT = 'Saved Aurora.'
const SENT_WITHIN = { timeout: LOOK_SEND_DELAY_MS + 2000 }

const getLooks = vi.mocked(apiClient.getThemeLooks)
const putLooks = vi.mocked(apiClient.putThemeLooks)

function Probe() {
  const { look, setLook } = useLook()
  return (
    <div>
      <p>Active: {look.name}</p>
      <button type="button" onClick={() => setLook(AURORA)}>Use Aurora</button>
      <LookNotice notice={{ ...storedNotice(SAVED_TEXT), seq: 1 }} />
    </div>
  )
}

function renderPlay(store: Storage) {
  return render(
    <UiStyleProvider initialStyle={BUILTIN_PLAY_STYLE}>
      <LookScopeContext.Provider value={SCOPE}>
        <LookProvider storage={() => store}>
          <Probe />
        </LookProvider>
      </LookScopeContext.Provider>
    </UiStyleProvider>,
  )
}

function seenStore(): Storage {
  const store = memoryStorage({ [KEYS.active]: JSON.stringify(DAYLIGHT) })
  setLookMark(() => store, SCOPE, BUILTIN_PLAY_LOOK_KEYS, 'synced', true)
  return store
}

beforeEach(() => {
  window.localStorage.clear()
  window.localStorage.setItem(APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION)
  clearRoot()
  vi.spyOn(console, 'warn').mockImplementation(() => undefined)
  getLooks.mockReset()
  putLooks.mockReset()
  getLooks.mockResolvedValue({ themeId: 'automatos/play', looks: [], active: DAYLIGHT, defaultLook: null, dropped: 0 } as never)
})

afterEach(() => {
  vi.restoreAllMocks()
  clearRoot()
})

describe("the LookProvider on the server's looks", () => {
  it("reads builtin Play's looks under its package id and writes a change through", async () => {
    putLooks.mockImplementation((_themeId, body) => Promise.resolve(body as never))
    const store = seenStore()
    renderPlay(store)
    await waitFor(() => expect(getLooks).toHaveBeenCalledWith('automatos/play'))
    act(() => screen.getByRole('button', { name: 'Use Aurora' }).click())
    expect(screen.getByText('Active: Aurora')).toBeInTheDocument()
    await waitFor(() => expect(putLooks).toHaveBeenCalledWith('automatos/play', { looks: [], active: { ...AURORA } }), SENT_WITHIN)
    await waitFor(() => expect(hasLookMark(() => store, SCOPE, BUILTIN_PLAY_LOOK_KEYS, 'pending')).toBe(false))
    expect(screen.getByText(SAVED_TEXT)).toBeInTheDocument()
  })

  it('shows the not-kept notice when the server refuses the save', async () => {
    putLooks.mockRejectedValue(Object.assign(new Error('That look does not fit.'), { status: 400 }))
    renderPlay(seenStore())
    await waitFor(() => expect(getLooks).toHaveBeenCalled())
    expect(screen.getByText(SAVED_TEXT)).toBeInTheDocument()
    act(() => screen.getByRole('button', { name: 'Use Aurora' }).click())
    expect(await screen.findByText(`${SAVED_TEXT} ${LOOKS_NOT_KEPT}`, {}, SENT_WITHIN)).toBeInTheDocument()
  })

  it("keeps quiet when the save just didn't reach the server: this browser keeps it, marked to send", async () => {
    putLooks.mockRejectedValue(new TypeError('Failed to fetch'))
    const store = seenStore()
    renderPlay(store)
    await waitFor(() => expect(getLooks).toHaveBeenCalled())
    act(() => screen.getByRole('button', { name: 'Use Aurora' }).click())
    await waitFor(() => expect(putLooks).toHaveBeenCalled(), SENT_WITHIN)
    await act(async () => undefined)
    expect(screen.getByText(SAVED_TEXT)).toBeInTheDocument()
    expect(hasLookMark(() => store, SCOPE, BUILTIN_PLAY_LOOK_KEYS, 'pending')).toBe(true)
    expect(JSON.parse(store.getItem(KEYS.active) ?? 'null')).toEqual(AURORA)
  })
})
