/**
 * PRD-244 — the chat shell and the Assignments hub carry their rules on their
 * own roots (`.sh-chat`, `.cc-page`) rather than on `.studio` alone, so each
 * Studio surface is self-contained. (They render only in the Studio style —
 * two styles, two tones — so no Classic token values are needed or defined.)
 */
import { describe, it, expect } from 'vitest'
import { readFileSync, readdirSync } from 'fs'
import path from 'path'

const FRONTEND = path.resolve(__dirname, '..', '..', '..')
const css = readFileSync(path.join(FRONTEND, 'app', 'globals.css'), 'utf8')

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
    expect(studioOnly(classesUsed(path.join(FRONTEND, 'components', 'chatbot'), /\bsh-chat-[a-z0-9-]+/g))).toEqual([])
  })

  it('the Assignments hub hangs off .cc-page like the Command Centre', () => {
    const hub = path.join(FRONTEND, 'components', 'assignments', 'studio')
    const used = classesUsed(hub, /\b(?:mis|pb|entry|mkt|status|act|pg|scope|l|view|cc)-[a-z0-9-]+/g)
    expect(used.size).toBeGreaterThan(10)
    expect(studioOnly(used)).toEqual([])
    expect(readFileSync(path.join(hub, 'assignments-hub.tsx'), 'utf8')).toContain('className="cc-page"')
  })

  it('no Classic value is given to a Studio-only token — nothing crosses between the styles', () => {
    const root = css.slice(css.indexOf(':root {'), css.indexOf('\n  }', css.indexOf(':root {')))
    for (const t of ['--fg', '--paper', '--navy', '--serif', '--cc-hover']) {
      expect(root, `${t} on :root`).not.toMatch(new RegExp(`^\\s*${t}:`, 'm'))
    }
  })
})
