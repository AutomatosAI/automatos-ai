/**
 * PRD-244 W3 — the Studio detector reads the Style axis (Classic | Studio),
 * never the tone. It is stable across SSR (the provider is seeded from the
 * cookie) and Classic outside the provider.
 */
import { describe, it, expect, vi } from 'vitest'
import { renderHook } from '@testing-library/react'
import type { ReactNode } from 'react'

vi.mock('next-themes', () => ({ useTheme: () => ({ theme: 'dark', resolvedTheme: 'dark', setTheme: vi.fn() }) }))

import { UiStyleProvider } from '@/contexts/ui-style-context'
import * as hook from '@/hooks/use-studio-theme'

const wrap = (style: 'classic' | 'studio') => ({ children }: { children: ReactNode }) => (
  <UiStyleProvider initialStyle={style}>{children}</UiStyleProvider>
)

describe('useIsStudio', () => {
  it('is true when the style is Studio, whatever the tone', () => {
    const { result } = renderHook(() => hook.useIsStudio(), { wrapper: wrap('studio') })
    expect(result.current).toBe(true)
  })

  it('is false for the Classic style', () => {
    const { result } = renderHook(() => hook.useIsStudio(), { wrapper: wrap('classic') })
    expect(result.current).toBe(false)
  })

  it('is Classic outside the provider', () => {
    const { result } = renderHook(() => hook.useIsStudio())
    expect(result.current).toBe(false)
  })

  it('no longer exports the URL flag hook', () => {
    expect(Object.keys(hook)).not.toContain('useStudioThemeFlag')
  })
})
