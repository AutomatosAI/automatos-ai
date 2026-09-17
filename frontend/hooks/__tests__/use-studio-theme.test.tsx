/**
 * PRD-244 W0 — the Studio detector. Until now nothing tested the hook every
 * theme fork depends on; the URL flag that used to sit beside it is gone.
 */
import { describe, it, expect, vi } from 'vitest'
import { renderHook } from '@testing-library/react'

const themeRef = vi.hoisted(() => ({ theme: 'dark', resolvedTheme: 'dark' }))
vi.mock('next-themes', () => ({ useTheme: () => themeRef }))

import * as hook from '@/hooks/use-studio-theme'

describe('useIsStudio', () => {
  it('is true once mounted when the picked theme is studio', () => {
    themeRef.theme = 'studio'
    themeRef.resolvedTheme = 'studio'
    const { result } = renderHook(() => hook.useIsStudio())
    expect(result.current).toBe(true)
  })

  it('is false for every other theme', () => {
    for (const t of ['light', 'dark', 'system']) {
      themeRef.theme = t
      themeRef.resolvedTheme = t === 'system' ? 'dark' : t
      const { result } = renderHook(() => hook.useIsStudio())
      expect(result.current).toBe(false)
    }
  })

  it('honours a resolved studio theme', () => {
    themeRef.theme = 'system'
    themeRef.resolvedTheme = 'studio'
    const { result } = renderHook(() => hook.useIsStudio())
    expect(result.current).toBe(true)
  })

  it('no longer exports the URL flag hook', () => {
    expect(Object.keys(hook)).not.toContain('useStudioThemeFlag')
  })
})
