/**
 * PRD-244 W2 (D3) — the chat shell and the Assignments hub render styled under
 * every theme: their rules hang off their own roots (`.sh-chat`, `.cc-page`),
 * never `.studio` alone, and the Studio design tokens have Light/Dark values.
 */
import { describe, it, expect } from 'vitest'
import { readFileSync, readdirSync } from 'fs'
import path from 'path'

const FRONTEND = path.resolve(__dirname, '..', '..', '..')
const css = readFileSync(path.join(FRONTEND, 'app', 'globals.css'), 'utf8')
const rootBlock = css.slice(css.indexOf(':root {'), css.indexOf('\n  }', css.indexOf(':root {')))

function classesUsed(dir: string, prefix: RegExp): Set<string> {
  const used = new Set<string>()
  for (const f of readdirSync(dir)) {
    if (!f.endsWith('.tsx')) continue
    for (const m of readFileSync(path.join(dir, f), 'utf8').matchAll(prefix)) used.add(m[0])
  }
  return used
}
function studioOnly(used: Set<string>): string[] {
  const out: string[] = []
  for (const cls of used) {
    const lines = css.split('\n').filter((l) => /\{/.test(l) && new RegExp(`\\.${cls}(?![a-z0-9-])`).test(l.split('{')[0]))
    if (lines.length && lines.every((l) => /^\s*\.studio\s/.test(l))) out.push(cls)
  }
  return out
}

describe('Studio surfaces outside .studio', () => {
  it('the chat shell hangs off .sh-chat', () => {
    expect(css).not.toMatch(/^\s*\.studio\s+\.sh-chat/m)
    expect(css).toMatch(/^\s*\.sh-chat \{/m)
    expect(css).toMatch(/^\s*html:not\(\.studio\) \.sh-chat \{ height:/m)
    expect(studioOnly(classesUsed(path.join(FRONTEND, 'components', 'chatbot'), /\bsh-chat-[a-z0-9-]+/g))).toEqual([])
  })

  it('the Assignments hub hangs off .cc-page like the Command Centre', () => {
    const hub = path.join(FRONTEND, 'components', 'assignments', 'studio')
    const used = classesUsed(hub, /\b(?:mis|pb|entry|mkt|status|act|pg|scope|l|view|cc)-[a-z0-9-]+/g)
    expect(used.size).toBeGreaterThan(10)
    expect(studioOnly(used)).toEqual([])
    expect(readFileSync(path.join(hub, 'assignments-hub.tsx'), 'utf8')).toContain('className="cc-page"')
  })

  it('every Studio-only design token has a Light/Dark value on :root', () => {
    for (const t of ['--fg', '--fg-2', '--fg-3', '--paper', '--bg-1', '--bg-2', '--border-2', '--navy', '--accent-2', '--serif', '--cc-hover', '--cc-hover-border', '--cc-live-dot']) {
      expect(rootBlock, `${t} on :root`).toMatch(new RegExp(`^\\s*${t}:`, 'm'))
    }
  })
})
