/**
 * PRD-261 A-19d1 — the generic look store's default server is the backend, through apiClient's named
 * theme methods (GET/PUT /api/themes/looks/<themeId>; lib/__tests__/api-client-error-body.test.ts
 * holds their paths): with no server given, the store reads the theme's looks back from
 * getThemeLooks(themeId) and writes a change through putThemeLooks(themeId, {looks, active}). So does
 * the look provider as the host mounts it (components/theme-host/theme-look-runtime.tsx), which hands
 * the store no server of its own.
 */
import { act, cleanup, render, renderHook, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { apiClient, type ThemeLooks } from '@/lib/api-client'
import { lookScope } from '@/lib/look-kit/scope'
import { ThemeLookProvider } from '@/lib/theme-host/looks/provider'
import { LOOK_SEND_DELAY_MS } from '@/lib/theme-host/looks/server'
import { useLookStore } from '@/lib/theme-host/looks/store'
import { hasLookMark, lookKeys, setLookMark, type StorageSource } from '@/lib/theme-host/looks/storage'
import { useLook, type HostLookState } from '@/lib/theme-sdk/looks'

import { DUSK, HELLO, HELLO_THEME_ID, helloThemeBoot, INK, memoryStorage, myLook, putPageBoot } from './hello-looks'

vi.mock('@/lib/api-client', () => {
  const client = { getThemeLooks: vi.fn(), putThemeLooks: vi.fn() }
  return { apiClient: client, default: client }
})

const SCOPE = lookScope('7d3c2a10-1b2c-4d5e-8f90-123456789abc', 'local')
const KEYS = lookKeys(SCOPE, HELLO.keys)
const SENT_WITHIN = { timeout: LOOK_SEND_DELAY_MS + 2000 }

const getLooks = vi.mocked(apiClient.getThemeLooks)
const putLooks = vi.mocked(apiClient.putThemeLooks)

/** What useLook() last handed the theme below the provider. */
const seen: { current: HostLookState | null } = { current: null }

function Probe() {
  const state = useLook()
  seen.current = state
  return <p>Looks: {state.look?.id ?? 'none'}</p>
}

function looksAnswer(fields: Partial<ThemeLooks> = {}): ThemeLooks {
  return { themeId: HELLO_THEME_ID, looks: [{ ...myLook() }], active: { ...INK }, defaultLook: null, dropped: 0, ...fields }
}

beforeEach(() => {
  vi.spyOn(console, 'warn').mockImplementation(() => undefined)
  getLooks.mockReset()
  putLooks.mockReset()
})

let removeBoot: () => void = () => undefined

afterEach(() => {
  cleanup()
  removeBoot()
  removeBoot = () => undefined
  seen.current = null
  vi.restoreAllMocks()
  document.documentElement.removeAttribute('style')
})

describe('the default server', () => {
  it("reads the theme's looks back from GET /api/themes/looks/<id> and writes a change through PUT", async () => {
    const store = memoryStorage({ [KEYS.active]: JSON.stringify(DUSK) })
    setLookMark(() => store, SCOPE, HELLO.keys, 'synced', true)
    getLooks.mockResolvedValue(looksAnswer())
    putLooks.mockImplementation((_themeId, body) => Promise.resolve(looksAnswer({ looks: body.looks, active: body.active })))
    const storage: StorageSource = () => store
    const { result } = renderHook(() => useLookStore(storage, SCOPE, HELLO))
    await waitFor(() => expect(result.current.look).toEqual(INK))
    expect(getLooks).toHaveBeenCalledWith(HELLO_THEME_ID)
    expect(result.current.saved).toEqual([myLook()])
    act(() => void result.current.setLook(DUSK))
    await waitFor(() => expect(putLooks).toHaveBeenCalledWith(HELLO_THEME_ID, { looks: [myLook()], active: DUSK }), SENT_WITHIN)
    expect(getLooks).toHaveBeenCalledTimes(1)
  })
})

describe('the look provider as the host mounts it (no server given)', () => {
  it("reads the theme's looks through GET and writes a change back through PUT, then clears the pending mark", async () => {
    removeBoot = putPageBoot(helloThemeBoot())
    const store = memoryStorage()
    const storage: StorageSource = () => store
    setLookMark(storage, SCOPE, HELLO.keys, 'synced', true)
    getLooks.mockResolvedValue(looksAnswer())
    putLooks.mockImplementation((_themeId, body) => Promise.resolve(looksAnswer({ looks: body.looks, active: body.active })))
    render(
      <ThemeLookProvider themeId={HELLO_THEME_ID} enabled tonePolicy="from-look" scope={SCOPE} storage={storage}>
        <Probe />
      </ThemeLookProvider>,
    )
    expect(await screen.findByText(`Looks: ${INK.id}`)).toBeInTheDocument()
    expect(getLooks).toHaveBeenCalledWith(HELLO_THEME_ID)
    act(() => void seen.current?.setLook(DUSK))
    await waitFor(() => expect(putLooks).toHaveBeenCalledWith(HELLO_THEME_ID, { looks: [myLook()], active: DUSK }), SENT_WITHIN)
    await waitFor(() => expect(hasLookMark(storage, SCOPE, HELLO.keys, 'pending')).toBe(false))
  })
})
