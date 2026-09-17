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
  it('the picker offers Style (Classic | Studio) and Tone (Light | Dark | System) — no Matte, no "preview"', () => {
    const src = read('components/ui/theme-toggle.tsx')
    expect(src).not.toMatch(/matte/i)
    expect(src).not.toMatch(/preview/i)
    expect(src).toContain('value="classic"')
    expect(src).toContain('value="studio"')
    expect(src).not.toContain("setTheme('studio')") // W3: Studio is a style, not a tone
  })

  it('providers list the two tones only, mount the style provider, and mount no URL flag', () => {
    const src = read('components/providers.tsx')
    expect(src).toContain("themes={['light', 'dark']}")
    expect(src).toContain('<UiStyleProvider initialStyle={initialUiStyle}>')
    expect(src).not.toContain('StudioThemeFlag')
    expect(src).toContain('defaultTheme="system"')
  })

  it('the stylesheet carries no Matte rules', () => {
    expect(read('app/globals.css')).not.toContain('.matte')
  })

  it('the chat page forks on the shared tablet breakpoint', () => {
    const src = read('app/chat/page.tsx')
    expect(src).toContain('if (isStudio && !isTabletOrBelow)')
    expect(src).not.toContain('if (isStudio && !isMobile)')
  })
})
