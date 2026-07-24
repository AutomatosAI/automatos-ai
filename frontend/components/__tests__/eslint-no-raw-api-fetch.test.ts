import { describe, it, expect } from 'vitest'
import { readFileSync } from 'fs'
import path from 'path'
// @ts-expect-error eslint ships no bundled types and @types/eslint is not a dependency
import { Linter } from 'eslint'

type LintMessage = { ruleId: string | null }

// PRD-154 S10 — components must call the platform apiClient, not raw
// fetch('/api…'). Raw fetch bypasses the base URL + auth headers and 404s in
// split-origin deploys (the exact bug S10/S12 fix). This rule (in the real
// .eslintrc.json) is exercised against fixtures via the eslint Linter API.

const eslintrc = JSON.parse(
  readFileSync(path.resolve(__dirname, '..', '..', '.eslintrc.json'), 'utf8'),
)
const linter = new Linter()
const rules = { 'no-restricted-syntax': eslintrc.rules['no-restricted-syntax'] }
const opts = { parserOptions: { ecmaVersion: 2021, sourceType: 'module' }, rules } as any

const lint = (code: string): LintMessage[] => linter.verify(code, opts)

describe("S10 eslint — ban raw fetch('/api…')", () => {
  it("flags fetch('/api/...')", () => {
    const msgs = lint(`fetch('/api/agents?search=x')`)
    expect(msgs.some((m) => m.ruleId === 'no-restricted-syntax')).toBe(true)
  })

  it('allows apiClient.request on /api', () => {
    const msgs = lint(`apiClient.request('/api/agents')`)
    expect(msgs.length).toBe(0)
  })

  it('allows fetch to a non-/api URL', () => {
    const msgs = lint(`fetch('https://example.com/x')`)
    expect(msgs.length).toBe(0)
  })
})
