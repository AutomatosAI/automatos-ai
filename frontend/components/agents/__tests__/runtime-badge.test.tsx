/**
 * The roster card names the runtime an agent's tickets run on.
 *
 * 2026-09-17: a Claude Code agent's card still read "GPT" — the API model
 * badge, which the runtime section itself says does not apply to sessions.
 */
import { describe, it, expect } from 'vitest'

import { runtimeBadge, type CliRegistryEntry } from '@/components/agents/runtime-section'

const registry: CliRegistryEntry[] = [
  { id: 'claude', label: 'Claude Code', model_hint: '', model_placeholder: '' },
  { id: 'codex', label: 'Codex', model_hint: '', model_placeholder: '' },
]

describe('runtimeBadge', () => {
  it('is null for an api agent, so the model badge stands', () => {
    expect(runtimeBadge({ runtime: 'api' }, registry)).toBeNull()
    expect(runtimeBadge(null, registry)).toBeNull()
    expect(runtimeBadge({}, registry)).toBeNull()
  })

  it('names the CLI from the registry and the pinned model', () => {
    expect(runtimeBadge({ runtime: 'cli', provider: 'claude', model: 'fable' }, registry)).toEqual({
      cli: 'Claude Code',
      model: 'fable',
      text: 'Claude Code · fable',
    })
  })

  it('shows just the CLI when no model is pinned', () => {
    expect(runtimeBadge({ runtime: 'cli', provider: 'codex', model: null }, registry)?.text).toBe('Codex')
    expect(runtimeBadge({ runtime: 'cli', provider: 'codex', model: '   ' }, registry)?.text).toBe('Codex')
  })

  it('falls back to the id, title-cased, until the registry has answered', () => {
    expect(runtimeBadge({ runtime: 'cli', provider: 'claude', model: 'opus' }, null)?.text).toBe('Claude · opus')
    expect(runtimeBadge({ runtime: 'cli', provider: 'some-new-cli', model: '' }, undefined)?.text).toBe('Some New Cli')
  })

  it('prefers the registry label over the id whenever the id is known', () => {
    expect(runtimeBadge({ runtime: 'cli', provider: 'codex', model: 'gpt-5.5' }, registry)?.cli).toBe('Codex')
    expect(runtimeBadge({ runtime: 'cli', provider: 'other', model: '' }, registry)?.cli).toBe('Other')
  })
})
