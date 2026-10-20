/**
 * PRD-261 A-13c — SDK area `looks` (lib/theme-sdk/looks.ts) against the seam the host's look runtime fills
 * (internal/look-runtime.ts). Outside a runtime every name answers "no looks here" and none throws; inside one,
 * useLook reads the runtime through context and the plain functions reach the runtime registered now, which a
 * stale runtime's cleanup never drops. The host's real runtime is proved by
 * components/theme-host/__tests__/theme-look-runtime.test.tsx. useLookScope reads the scope the host's
 * AccountLookScope provides (components/looks/__tests__/look-scope.test.tsx runs the real one under Play's shelf).
 */
import { renderHook } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { LookScopeContext } from '@/components/theme-host/look-scope'
import { DEFAULT_LOOK_SCOPE, lookScope } from '@/lib/looks/scope'

import { LookRuntimeContext, currentLookRuntime, registerLookRuntime, type HostLookRuntime } from '../internal/look-runtime'
import { exportLook, importLook, listLooks, saveLook, useLook, useLookScope, type HostLook, type HostLookState } from '../looks'

const NOT_AVAILABLE = { ok: false, errors: ["Looks aren't available here."] }
const LOOK: HostLook = { schema: 'automatos-interface-look/v1', id: 'look-1', name: 'Mine', sub: 'Your look' }
const FILE = { fileName: 'Mine.look.json', text: '{}\n', type: 'application/json' }

function fakeRuntime(name: string): HostLookRuntime {
  const state: HostLookState = {
    available: true,
    look: LOOK,
    saved: [{ ...LOOK, name }],
    loaded: true,
    persisted: true,
    tone: 'light',
    setLook: vi.fn(() => ({ ok: true as const, look: LOOK })),
    deleteLook: vi.fn(),
    renameLook: vi.fn(() => ({ ok: true as const, look: LOOK })),
  }
  return {
    state,
    saveLook: vi.fn(() => ({ ok: true as const, look: LOOK })),
    exportLook: vi.fn(() => FILE),
    importLook: vi.fn(() => ({ ok: true as const, look: { ...LOOK, id: 'look-2' } })),
  }
}

const cleanups: Array<() => void> = []

afterEach(() => {
  cleanups.splice(0).forEach((cleanup) => cleanup())
})

function register(runtime: HostLookRuntime): () => void {
  const cleanup = registerLookRuntime(runtime)
  cleanups.push(cleanup)
  return cleanup
}

describe('PRD-261 A-13c — looks outside a look runtime (Classic, Studio, a theme without looks)', () => {
  it('useLook answers no looks: no look, nothing saved, frozen, and its actions refuse', () => {
    const { result } = renderHook(() => useLook())
    expect(result.current).toMatchObject({ available: false, look: null, saved: [], loaded: true, persisted: false, tone: null })
    expect(Object.isFrozen(result.current)).toBe(true)
    expect(result.current.setLook(LOOK)).toEqual(NOT_AVAILABLE)
    expect(result.current.renameLook('look-1', 'Other')).toEqual(NOT_AVAILABLE)
    expect(() => result.current.deleteLook('look-1')).not.toThrow()
  })

  it('the plain functions refuse in plain words and never throw', () => {
    expect(currentLookRuntime()).toBeNull()
    expect(saveLook(LOOK)).toEqual(NOT_AVAILABLE)
    expect(importLook('{}')).toEqual(NOT_AVAILABLE)
    expect(listLooks()).toEqual([])
    expect(exportLook(LOOK)).toBeNull()
  })
})

describe('PRD-261 A-13c — looks on a mounted look runtime', () => {
  it("useLook reads the runtime the host put in context, the theme re-rendering on the runtime's state", () => {
    const runtime = fakeRuntime('Mine')
    const wrapper = ({ children }: { children: ReactNode }) => (
      <LookRuntimeContext.Provider value={runtime}>{children}</LookRuntimeContext.Provider>
    )
    const { result } = renderHook(() => useLook(), { wrapper })
    expect(result.current).toBe(runtime.state)
  })

  it('the plain functions reach the runtime registered now, and hand it what the theme gave', () => {
    const runtime = fakeRuntime('Mine')
    register(runtime)
    expect(saveLook(LOOK)).toEqual({ ok: true, look: LOOK })
    expect(runtime.saveLook).toHaveBeenCalledWith(LOOK)
    expect(listLooks().map((look) => look.name)).toEqual(['Mine'])
    expect(exportLook(LOOK)).toEqual(FILE)
    expect(importLook('{"name":"Mine"}')).toMatchObject({ ok: true, look: { id: 'look-2' } })
    expect(runtime.importLook).toHaveBeenCalledWith('{"name":"Mine"}')
  })

  it("a replaced runtime's cleanup never drops the one that replaced it; the current one's does", () => {
    const first = fakeRuntime('First')
    const second = fakeRuntime('Second')
    const dropFirst = register(first)
    const dropSecond = register(second)
    dropFirst()
    expect(currentLookRuntime()).toBe(second)
    expect(listLooks().map((look) => look.name)).toEqual(['Second'])
    dropSecond()
    expect(currentLookRuntime()).toBeNull()
    expect(saveLook(LOOK)).toEqual(NOT_AVAILABLE)
  })
})

describe("PRD-261 W-A5 — useLookScope, whose looks these are", () => {
  it("reads the scope the host's AccountLookScope provides, and null while the account isn't known", () => {
    for (const scope of [lookScope('ws-1', 'user_a'), null]) {
      const wrapper = ({ children }: { children: ReactNode }) => (
        <LookScopeContext.Provider value={scope}>{children}</LookScopeContext.Provider>
      )
      expect(renderHook(() => useLookScope(), { wrapper }).result.current).toBe(scope)
    }
  })

  it("answers the host's default scope outside it (an isolated mount)", () => {
    expect(renderHook(() => useLookScope()).result.current).toBe(DEFAULT_LOOK_SCOPE)
  })
})
