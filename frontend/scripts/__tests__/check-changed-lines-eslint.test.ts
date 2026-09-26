// @vitest-environment node
/**
 * The changed-lines ESLint gate (scripts/check-changed-lines-eslint.js): which findings
 * count. Only the enforced rules count, and only on the change's own lines. A capitalised
 * component gets the component allowance, and test files are exempt from length and depth.
 */
import { createRequire } from 'module'
import { describe, expect, it } from 'vitest'

const require = createRequire(import.meta.url)
const gate = require('../check-changed-lines-eslint.js')

const longFunction = (name: string, lines: number, line = 10, endLine = 80) => ({
  ruleId: 'max-lines-per-function',
  message: `Function '${name}' has too many lines (${lines}). Maximum allowed is 50.`,
  line,
  endLine,
})

describe('changedLines', () => {
  it('reads the new side of each hunk and skips deleted files', () => {
    const diff = [
      '+++ b/frontend/a.tsx',
      '@@ -10,0 +11,3 @@',
      '@@ -40 +44 @@',
      '+++ /dev/null',
      '@@ -1,5 +0,0 @@',
    ].join('\n')
    const result = gate.changedLines(diff)
    expect([...result.keys()]).toEqual(['frontend/a.tsx'])
    expect([...result.get('frontend/a.tsx')]).toEqual([11, 12, 13, 44])
  })
})

describe('isFinding', () => {
  const touched = new Set([20])

  it('counts a long helper function the change touches', () => {
    expect(gate.isFinding(longFunction('formatRow', 73), 'frontend/lib/x.ts', touched)).toBe(true)
  })

  it('ignores a long function the change does not touch', () => {
    expect(gate.isFinding(longFunction('formatRow', 73, 100, 180), 'frontend/lib/x.ts', touched)).toBe(false)
  })

  it('allows a React component up to the component limit, and no further', () => {
    expect(gate.isFinding(longFunction('SocialsTab', 140), 'frontend/components/x.tsx', touched)).toBe(false)
    expect(gate.isFinding(longFunction('SocialsTab', 190), 'frontend/components/x.tsx', touched)).toBe(true)
  })

  it('exempts test files from length and depth, but not from the apiClient rule', () => {
    const rawFetch = { ruleId: 'no-restricted-syntax', message: 'Use apiClient instead of raw fetch', line: 20, endLine: 20 }
    expect(gate.isFinding(longFunction('suite', 300), 'frontend/components/__tests__/x.test.tsx', touched)).toBe(false)
    expect(gate.isFinding(rawFetch, 'frontend/components/__tests__/x.test.tsx', touched)).toBe(true)
  })

  it('ignores rules this gate does not enforce', () => {
    const other = { ruleId: 'react-hooks/exhaustive-deps', message: 'missing dep', line: 20, endLine: 20 }
    expect(gate.isFinding(other, 'frontend/components/x.tsx', touched)).toBe(false)
  })
})

describe('fileSizeFinding', () => {
  it('fails a new file over the limit and warns when an existing big file grows', () => {
    expect(gate.fileSizeFinding('frontend/new.tsx', true, 0, 801).level).toBe('error')
    expect(gate.fileSizeFinding('frontend/old.ts', false, 900, 910).level).toBe('warning')
    expect(gate.fileSizeFinding('frontend/old.ts', false, 910, 900)).toBeNull()
    expect(gate.fileSizeFinding('frontend/small.ts', true, 0, 800)).toBeNull()
  })
})
