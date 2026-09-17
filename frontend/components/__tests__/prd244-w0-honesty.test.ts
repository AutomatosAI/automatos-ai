/**
 * PRD-244 W0 — grep gates (same pattern as studio-honest-chrome): Matte and the
 * "preview" label are gone from the picker, the URL flag is gone, the chat page
 * forks at the shared 1024 px breakpoint.
 */
import { describe, it, expect } from 'vitest'
import { readFileSync } from 'fs'
import path from 'path'

const ROOT = path.resolve(__dirname, '..', '..')
const read = (rel: string) => readFileSync(path.join(ROOT, rel), 'utf8')

describe('PRD-244 W0 honesty', () => {
  it('the picker offers Light / Dark / Studio / System — no Matte, no "preview"', () => {
    const src = read('components/ui/theme-toggle.tsx')
    expect(src).not.toMatch(/matte/i)
    expect(src).not.toMatch(/preview/i)
    expect(src).toContain("setTheme('studio')")
  })

  it('providers list light, dark and studio only, and mount no URL flag', () => {
    const src = read('components/providers.tsx')
    expect(src).toContain("themes={['light', 'dark', 'studio']}")
    expect(src).not.toContain('StudioThemeFlag')
    expect(src).toContain('defaultTheme="system"') // D1b flips this only after the Wave-4 pass
  })

  it('the stylesheet carries no Matte rules', () => {
    expect(read('app/globals.css')).not.toContain('.matte')
  })

  it('the chat page forks on the shared tablet breakpoint — and, since Wave 2, on width alone', () => {
    const src = read('app/chat/page.tsx')
    expect(src).toContain('if (!isTabletOrBelow)')
    expect(src).not.toContain('isStudio')
    expect(src).not.toContain('if (isStudio && !isMobile)')
  })
})
