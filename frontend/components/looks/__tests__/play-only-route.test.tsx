/**
 * PRD-261 W-A2 review (M3) — 'Switch to Play' is the theme menu's pick: it goes through
 * useChooseStyle (stored on the server, refused when members may not choose, put back on refusal),
 * and once the workspace installed Play's own package (D1-c hides builtin Play) it picks that install.
 */
import { describe, expect, it } from 'vitest'
import { readFileSync } from 'fs'
import path from 'path'

import { playStyleFor } from '@/components/looks/play-only-route'
import type { InstalledTheme } from '@/lib/api-client'
import { BUILTIN_PLAY_STYLE } from '@/lib/theme-host/builtin/play-entry'

const src = readFileSync(path.resolve(__dirname, '..', 'play-only-route.tsx'), 'utf8')

function install(overrides: Partial<InstalledTheme>): InstalledTheme {
  return {
    installId: 'i-1', themeId: 'automatos/play', name: 'Play', packageId: 'p-1', version: '1.0.0',
    ownerType: 'first_party', reviewed: true, private: false, isDefault: false,
    pinnedVersion: null, previousPackageId: null, pendingPackageId: null, pendingVersion: null,
    installedAt: null, optIn: { required: false, wording: '', packageId: 'p-1' }, revoked: null, needsUpdate: false,
    ...overrides,
  }
}

describe('Switch to Play', () => {
  it('is stored through useChooseStyle, never a bare setStyle', () => {
    expect(src).toContain('const chooseStyle = useChooseStyle()')
    expect(src).toContain('chooseStyle(playStyleFor(uiStyle.installedThemes))')
    expect(src).not.toContain('uiStyle.setStyle(')
  })

  it('picks builtin Play when the workspace has no install of its package', () => {
    expect(playStyleFor([])).toBe(BUILTIN_PLAY_STYLE)
    expect(playStyleFor([install({ themeId: 'acme/neon' })])).toBe(BUILTIN_PLAY_STYLE)
  })

  it("picks the install of Play's own package, which hides builtin Play (D1-c)", () => {
    expect(playStyleFor([install({ installId: 'abc' })])).toBe('theme:install:abc')
  })

  it('ignores a withdrawn install, as the menu does', () => {
    const withdrawn = install({ revoked: { level: 'disable', reason: 'withdrawn' } })
    expect(playStyleFor([withdrawn])).toBe(BUILTIN_PLAY_STYLE)
  })
})
