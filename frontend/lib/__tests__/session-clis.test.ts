/**
 * PRD-253 — the session CLIs by name, in step with the backend registry.
 */
import { describe, expect, it } from 'vitest'

import { SESSION_CLIS, sessionCli } from '@/lib/session-clis'

describe('session CLIs', () => {
  it('names the session CLIs exactly as the backend registry does', async () => {
    const fs = await import('node:fs')
    const path = await import('node:path')
    const src = fs.readFileSync(path.resolve(__dirname, '../../../orchestrator/core/cli_presets.py'), 'utf8')
    const ids = Object.fromEntries([...src.matchAll(/^(PROVIDER_[A-Z]+) = "([a-z]+)"$/gm)].map((m) => [m[1], m[2]]))
    const rows = [...src.matchAll(/(PROVIDER_[A-Z]+): CliPresetInfo\(\s*\1, "([^"]+)", "([^"]+)"/g)]
      .map((m) => [ids[m[1]], { label: m[2], usageSlug: m[3] }])
    expect(rows.length).toBeGreaterThan(0)
    expect(Object.fromEntries(rows)).toEqual(SESSION_CLIS)
  })

  it('reads an id as given, defaults to Claude Code and keeps an unknown CLI by its id', () => {
    expect(sessionCli(' Copilot ').label).toBe('GitHub Copilot')
    expect(sessionCli(null)).toEqual({ label: 'Claude Code', usageSlug: 'claude_code' })
    expect(sessionCli('grok')).toEqual({ label: 'grok', usageSlug: 'grok' })
  })
})
