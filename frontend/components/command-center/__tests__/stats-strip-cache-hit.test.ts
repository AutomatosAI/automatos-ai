/** PRD-244 review — CACHE-HIT is wired to the analytics read, never "metric pending". */
import { describe, it, expect } from 'vitest'
import { readFileSync } from 'fs'
import path from 'path'

const src = readFileSync(path.resolve(__dirname, '..', 'stats-strip.tsx'), 'utf8')

describe('StatsStrip CACHE-HIT', () => {
  it('reads cacheShare and cacheReadTokens from the unified analytics summary', () => {
    expect(src).toContain('cost?.summary?.cacheShare')
    expect(src).toContain('cost?.summary?.cacheReadTokens')
    expect(src).not.toContain('metric pending')
  })
  it('the honest empties are named', () => {
    expect(src).toContain("'no cached reads · 7d'")
    expect(src).toContain("'no requests · 7d'")
  })
})
