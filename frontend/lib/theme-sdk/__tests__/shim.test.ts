/**
 * PRD-261 A-10a — the SDK's shim (/theme-sdk/v1/sdk.js, written by
 * scripts/theme-sdk/generate-shims.js) exports every surface name. Its names are the union of
 * lib/theme-sdk/surface/*.json, not a hand list in shared-modules.json; a name in two surface
 * files, or one that is not an identifier, stops the generator before it writes anything. sonner
 * is never shared: a theme reaches the toaster only through the SDK's notices.
 */
import { mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'fs'
import { createRequire } from 'module'
import os from 'os'
import path from 'path'
import { afterEach, describe, expect, it } from 'vitest'

import sharedModules from '@/lib/theme-host/shared-modules.json'

interface ShimEntry {
  readonly specifier: string
  readonly file: string
  readonly names: string[]
}

interface ShimGenerator {
  namesFromSurface(surfaceDir: string, frontendRoot: string): string[]
  sharedModuleExports(frontendRoot: string): ShimEntry[]
  renderShim(entry: { readonly specifier: string }, names: readonly string[], hostGlobal: string): string
}

interface SharedEntry {
  readonly specifier: string
  readonly file: string
  readonly exports?: unknown
  readonly surface?: string
}

const SDK_SPECIFIER = '@automatos/theme-sdk'
const load = createRequire(import.meta.url)
const generator = load('../../../scripts/theme-sdk/generate-shims.js') as ShimGenerator
const FRONTEND = path.resolve(__dirname, '../../..')
const SURFACE_DIR = path.join(FRONTEND, 'lib/theme-sdk/surface')
const SKIPPED = new Set(['default', '__esModule'])
const ENTRIES = sharedModules.modules as readonly SharedEntry[]

function surfaceUnion(): string[] {
  const names = readdirSync(SURFACE_DIR)
    .filter((file) => file.endsWith('.json'))
    .flatMap((file) => (JSON.parse(readFileSync(path.join(SURFACE_DIR, file), 'utf8')) as { names: string[] }).names)
  return Array.from(new Set(names)).sort()
}

function sdkShimEntry(): ShimEntry {
  const entry = generator.sharedModuleExports(FRONTEND).find((candidate) => candidate.specifier === SDK_SPECIFIER)
  if (!entry) throw new Error(`no shim entry for ${SDK_SPECIFIER}`)
  return entry
}

const scratch: string[] = []

/** A throwaway surface folder holding `files` (name → JSON body). */
function surfaceFolder(files: Readonly<Record<string, unknown>>): string {
  const dir = mkdtempSync(path.join(os.tmpdir(), 'theme-sdk-surface-'))
  scratch.push(dir)
  Object.entries(files).forEach(([name, body]) => writeFileSync(path.join(dir, name), JSON.stringify(body)))
  return dir
}

afterEach(() => {
  scratch.splice(0).forEach((dir) => rmSync(dir, { recursive: true, force: true }))
})

describe('PRD-261 A-10a — the SDK shim', () => {
  it('takes the SDK entry from its surface folder; the hand list is gone', () => {
    const entry = ENTRIES.find((candidate) => candidate.specifier === SDK_SPECIFIER)
    expect(entry?.surface).toBe('lib/theme-sdk/surface')
    expect(entry && 'exports' in entry).toBe(false)
  })

  it('exports every surface name, and nothing else', () => {
    expect(sdkShimEntry().names).toEqual(surfaceUnion())
  })

  it("exports every name the SDK module exports at runtime", async () => {
    const sdk = (await import('@automatos/theme-sdk')) as Record<string, unknown>
    expect(sdkShimEntry().names).toEqual(Object.keys(sdk).filter((name) => !SKIPPED.has(name)).sort())
  })

  it('writes each surface name into the shim text', () => {
    const entry = sdkShimEntry()
    const shim = generator.renderShim(entry, entry.names, sharedModules.hostGlobal)
    const written = Array.from(shim.matchAll(/^ {2}([A-Za-z_$][\w$]*),$/gm), (match) => match[1])
    expect(written).toEqual(surfaceUnion())
    expect(shim).toContain(`host.modules["${SDK_SPECIFIER}"]`)
  })

  it('never shares sonner: notices go through the SDK', () => {
    expect(ENTRIES.map((entry) => entry.specifier).filter((specifier) => specifier.includes('sonner'))).toEqual([])
  })
})

describe('PRD-261 A-10a — reading a surface folder', () => {
  it('returns the sorted union of every file', () => {
    const dir = surfaceFolder({ 'b.json': { names: ['useB', 'beta'] }, 'a.json': { names: ['useA'] }, 'notes.txt': 'skipped' })
    expect(generator.namesFromSurface(dir, FRONTEND)).toEqual(['beta', 'useA', 'useB'])
  })

  it('refuses a name planned in two areas', () => {
    const dir = surfaceFolder({ 'a.json': { names: ['notify'] }, 'b.json': { names: ['notify'] } })
    expect(() => generator.namesFromSurface(dir, FRONTEND)).toThrow('notify is in both a.json and b.json.')
  })

  it('refuses a name it could not write safely, and a file with no names', () => {
    expect(() => generator.namesFromSurface(surfaceFolder({ 'a.json': { names: ['a-b'] } }), FRONTEND)).toThrow()
    expect(() => generator.namesFromSurface(surfaceFolder({ 'a.json': { names: [] } }), FRONTEND)).toThrow('a.json lists no names.')
  })

  it('refuses an empty folder', () => {
    expect(() => generator.namesFromSurface(surfaceFolder({}), FRONTEND)).toThrow('holds no surface files.')
  })
})
