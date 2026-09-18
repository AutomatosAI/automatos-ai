/**
 * PRD-244 W3 — two styles × two tones, grep gates: the style is read on the
 * server, the tone stays with next-themes, Studio Dark defines every colour
 * token Studio Light defines, and no code tests the theme value for "studio".
 */
import { describe, it, expect } from 'vitest'
import { readFileSync, readdirSync, statSync } from 'fs'
import path from 'path'

const ROOT = path.resolve(__dirname, '..', '..')
const read = (rel: string) => readFileSync(path.join(ROOT, rel), 'utf8')
const css = read('app/globals.css')

function block(selector: string): string {
  const i = css.indexOf(`\n  ${selector} {\n`)
  expect(i, `${selector} block`).toBeGreaterThan(-1)
  return css.slice(i, css.indexOf('\n  }', i))
}
function colourTokens(body: string): string[] {
  return [...body.matchAll(/^\s+(--[a-z0-9-]+):\s*([^;]+);/gm)]
    .filter(([, , v]) => /^\d+ \d+% \d+%$/.test(v.trim()) || v.trim().startsWith('hsl('))
    .map(([, name]) => name)
}
function walk(dir: string, out: string[] = []): string[] {
  for (const f of readdirSync(dir)) {
    const p = path.join(dir, f)
    if (statSync(p).isDirectory()) { if (f !== '__tests__' && f !== 'node_modules') walk(p, out) }
    else if (/\.(tsx?|css)$/.test(f)) out.push(p)
  }
  return out
}

describe('PRD-244 W3 — Style × Tone', () => {
  it('the root layout reads the style cookie and puts the Studio class on <html> server-side', () => {
    const src = read('app/layout.tsx')
    expect(src).toContain("from 'next/headers'")
    expect(src).toContain('UI_STYLE_COOKIE')
    expect(src).toContain('STUDIO_HTML_CLASS')
    expect(src).toContain('initialUiStyle={uiStyle}')
  })

  it('the Studio detector reads the style context, not next-themes', () => {
    const src = read('hooks/use-studio-theme.ts')
    expect(src).not.toContain('next-themes')
    expect(src).toContain('useUiStyleOptional')
  })

  it('Tailwind dark variant keys off .dark only (no Matte)', () => {
    const cfg = readdirSync(ROOT).find((f) => f.startsWith('tailwind.config.'))!
    expect(read(cfg)).not.toContain('.matte')
  })

  it('Studio Dark defines every colour token Studio Light defines', () => {
    const light = colourTokens(block('.studio'))
    const dark = new Set(colourTokens(block('.studio.dark')))
    expect(light.length).toBeGreaterThan(40)
    const missing = light.filter((t) => !dark.has(t))
    expect(missing).toEqual([])
  })

  it('no source tests the theme value for "studio" any more', () => {
    const offenders = ['app', 'components', 'hooks', 'lib', 'contexts']
      .flatMap((d) => walk(path.join(ROOT, d)))
      .filter((f) => /theme === 'studio'|resolvedTheme === 'studio'|setTheme\('studio'\)/.test(readFileSync(f, 'utf8')))
      .map((f) => path.relative(ROOT, f))
    expect(offenders).toEqual([])
  })
})
