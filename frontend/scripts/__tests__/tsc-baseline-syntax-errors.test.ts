// @vitest-environment node
/**
 * The tsc baseline gate (scripts/tsc-baseline-check.js) fails on any syntax error.
 *
 * tsc reports syntax errors instead of type errors: a file that fails to parse stops
 * it before type checking, so the error count drops under the baseline and the ratchet
 * passed. PR #1121 went green with a stray backtick on line 1 of landing-page.tsx.
 */
import fs from 'fs'
import os from 'os'
import path from 'path'
import { createRequire } from 'module'
import { afterEach, describe, expect, it } from 'vitest'

const require = createRequire(import.meta.url)
const gate = require('../tsc-baseline-check.js')
const OPTIONS = { noEmit: true, jsx: 4 /* preserve */, skipLibCheck: true, noLib: true }

const dirs: string[] = []

function fileWith(source: string): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tsc-gate-'))
  dirs.push(dir)
  const file = path.join(dir, 'page.tsx')
  fs.writeFileSync(file, source)
  return file
}

afterEach(() => {
  for (const dir of dirs.splice(0)) fs.rmSync(dir, { recursive: true, force: true })
})

describe('syntaxErrors', () => {
  it('finds the stray backtick that broke landing-page.tsx', () => {
    const file = fileWith("`\n'use client'\n\nexport const title = `Command ${'Center'}`\nconst x = 1\n")
    const errors = gate.syntaxErrors([file], OPTIONS)
    expect(errors.length).toBeGreaterThan(0)
    expect(errors[0]).toContain('page.tsx')
  })

  it('finds nothing in a file that parses, type errors included', () => {
    const file = fileWith("'use client'\n\nexport const count: number = 'not a number'\n")
    expect(gate.syntaxErrors([file], OPTIONS)).toEqual([])
  })
})
