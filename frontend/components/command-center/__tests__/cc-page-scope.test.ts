/**
 * PRD-244 W1 (D2) — the Command Centre renders styled under every theme.
 *
 * Generalises calendar-tab-scope: every `cc-*` rule hangs off
 * `:is(.studio, .cc-page)` (or the shell root itself), never `.studio` alone,
 * so the shell — which is now the Command Centre at every desktop width —
 * is not a stack of unstyled divs under Light or Dark.
 */
import { describe, it, expect } from 'vitest'
import { readFileSync, readdirSync } from 'fs'
import path from 'path'

const FRONTEND = path.resolve(__dirname, '..', '..', '..')
const css = readFileSync(path.join(FRONTEND, 'app', 'globals.css'), 'utf8')

describe('Command Centre rules are reachable outside .studio', () => {
  it('no cc-* rule is scoped to .studio alone', () => {
    expect(css).not.toMatch(/^\s*\.studio\s+\.cc-/m)
    expect(css).not.toMatch(/,\s*\.studio\s+\.cc-/m)
  })

  it('the shell root and its tabs are scoped to the root', () => {
    expect(css).toMatch(/^\s*\.cc-page \{/m)
    expect(css).toMatch(/^\s*:is\(\.studio, \.cc-page\) \.cc-tabs \{/m)
    expect(css).toMatch(/^\s*:is\(\.studio, \.cc-page\) \.cc-body \{/m)
  })

  it('every cc-* class a Command Centre component uses is styled without .studio', () => {
    const dir = path.join(FRONTEND, 'components', 'command-center')
    const used = new Set<string>()
    for (const f of readdirSync(dir)) {
      if (!f.endsWith('.tsx')) continue
      for (const m of readFileSync(path.join(dir, f), 'utf8').matchAll(/\bcc-[a-z0-9-]+/g)) used.add(m[0])
    }
    expect(used.size).toBeGreaterThan(20)
    const studioOnly: string[] = []
    for (const cls of used) {
      const selectorLines = css.split('\n').filter((l) => /\{/.test(l) && new RegExp(`\\.${cls}(?![a-z0-9-])`).test(l.split('{')[0]))
      if (selectorLines.length === 0) continue // unstyled class — not this test's concern
      if (selectorLines.every((l) => /^\s*\.studio\s/.test(l))) studioOnly.push(cls)
    }
    expect(studioOnly).toEqual([])
  })
})
