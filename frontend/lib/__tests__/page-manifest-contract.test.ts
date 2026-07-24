import { describe, it, expect } from 'vitest'
import { readFileSync, existsSync } from 'fs'
import path from 'path'
import {
  PAGE_MANIFEST,
  resolvePageKey,
  getPageEntry,
} from '@/lib/generated/page-manifest'

// PRD-221 S6 — the generated mirror is a checked-in contract. These lock it to
// the backend source of truth (orchestrator/contracts/page-manifest.json) and
// to the real app router, so drift fails CI instead of shipping a stale mirror.

const CONTRACT = path.resolve(
  __dirname,
  '..',
  '..',
  '..',
  'orchestrator',
  'contracts',
  'page-manifest.json',
)
const APP_DIR = path.resolve(__dirname, '..', '..', 'app')

describe('page-manifest mirror ↔ backend contract', () => {
  it('mirror is in sync with the JSON contract', () => {
    const json = JSON.parse(readFileSync(CONTRACT, 'utf8'))
    // deep-equal is key-order insensitive, so the sorted-keys generator output
    // still matches the source array exactly.
    expect(PAGE_MANIFEST).toEqual(json.pages)
  })

  it('every manifest route resolves to an app-router directory', () => {
    for (const entry of PAGE_MANIFEST) {
      const seg = entry.route.replace(/^\//, '')
      const dir = path.resolve(APP_DIR, seg)
      expect(existsSync(dir), `${entry.route} → ${dir} must exist`).toBe(true)
    }
  })

  it('seeds exactly the 12 pages with unique kebab keys', () => {
    expect(PAGE_MANIFEST).toHaveLength(12)
    const keys = PAGE_MANIFEST.map((e) => e.key)
    expect(new Set(keys).size).toBe(12)
    for (const k of keys) expect(k).toMatch(/^[a-z][a-z0-9-]*$/)
  })
})

describe('resolvePageKey', () => {
  it('resolves exact routes and subpaths by longest prefix', () => {
    expect(resolvePageKey('/command-center')).toBe('command-center')
    expect(resolvePageKey('/missions/abc-123')).toBe('missions')
    expect(resolvePageKey('/chat/xyz')).toBe('chat')
    expect(resolvePageKey('/marketplace/widgets/5')).toBe('marketplace')
  })

  it('returns "unknown" for an unmapped route', () => {
    expect(resolvePageKey('/nope')).toBe('unknown')
    expect(resolvePageKey('/')).toBe('unknown')
  })
})

describe('getPageEntry', () => {
  it('returns the entry for a known key and undefined otherwise', () => {
    expect(getPageEntry('command-center')?.title).toBe('Command Centre')
    expect(getPageEntry('nope')).toBeUndefined()
  })
})
