/**
 * PRD-261 A-10a — the SDK's surface snapshot. Each area module exports exactly the names its
 * surface/<area>.json plans, the SDK module ('@automatos/theme-sdk', lib/theme-sdk/index.ts)
 * exports exactly their union, and no name sits in two areas. An export nobody planned, or a
 * planned name nobody exports, fails here. Stubbed areas throw "not in this SDK build" for every
 * name until their lane lands.
 */
import { readdirSync, readFileSync } from 'fs'
import path from 'path'
import { describe, expect, it } from 'vitest'

import { NOT_IN_THIS_BUILD } from '../internal/stub'

interface SurfaceFile {
  readonly area: string
  readonly story: string
  readonly status: 'built' | 'stub'
  readonly names: readonly string[]
}

type ModuleLoader = () => Promise<unknown>

const SDK_DIR = path.resolve(__dirname, '..')
const SURFACE_DIR = path.join(SDK_DIR, 'surface')
const IDENTIFIER = /^[A-Za-z_$][\w$]*$/
const SKIPPED = new Set(['default', '__esModule'])
const STATUSES = ['built', 'stub']
const BUILT_BY_THIS_STORY = ['theme', 'context', 'navigation', 'notices', 'events']
/** The stubbed areas whose lane has landed since (W-A4P A-12h: pages/command-center). */
const BUILT_BY_LATER_LANES = ['pages/command-center']

/** Every area and its module, written out so the bundler sees each import. */
const AREA_MODULES: Readonly<Record<string, ModuleLoader>> = {
  theme: () => import('../theme'),
  context: () => import('../context'),
  navigation: () => import('../navigation'),
  notices: () => import('../notices'),
  events: () => import('../events'),
  chat: () => import('../chat'),
  deliverables: () => import('../deliverables'),
  cards: () => import('../cards'),
  looks: () => import('../looks'),
  assets: () => import('../assets'),
  'pages/agents': () => import('../pages/agents'),
  'pages/assignments': () => import('../pages/assignments'),
  'pages/documents': () => import('../pages/documents'),
  'pages/playbooks': () => import('../pages/playbooks'),
  'pages/command-center': () => import('../pages/command-center'),
}

const SURFACE_FILES = readdirSync(SURFACE_DIR).filter((file) => file.endsWith('.json')).sort()
const SURFACES: readonly SurfaceFile[] = SURFACE_FILES.map(
  (file) => JSON.parse(readFileSync(path.join(SURFACE_DIR, file), 'utf8')) as SurfaceFile,
)
const PLANNED = SURFACES.flatMap((surface) => surface.names)

function runtimeNames(module: unknown): string[] {
  return Object.keys(module as Record<string, unknown>).filter((name) => !SKIPPED.has(name)).sort()
}

function areaCase(surface: SurfaceFile): [string, SurfaceFile] {
  return [surface.area, surface]
}

async function loadArea(area: string): Promise<Record<string, unknown>> {
  return (await AREA_MODULES[area]()) as Record<string, unknown>
}

/** What a module's exports have that the plan does not, and what the plan has that they lack. */
function surfaceDrift(exported: readonly string[], planned: readonly string[]): { unplanned: string[]; missing: string[] } {
  return {
    unplanned: exported.filter((name) => !planned.includes(name)),
    missing: planned.filter((name) => !exported.includes(name)),
  }
}

describe('PRD-261 A-10a — the surface files', () => {
  it('has one file per area module, and one module per file', () => {
    expect(SURFACES.map((surface) => surface.area).sort()).toEqual(Object.keys(AREA_MODULES).sort())
  })

  it('names each file after its area (pages/agents → pages-agents.json)', () => {
    expect(SURFACES.map((surface) => `${surface.area.replace('/', '-')}.json`)).toEqual(SURFACE_FILES)
  })

  it.each(SURFACES.map(areaCase))('%s lists sorted, distinct identifiers', (_area, surface) => {
    expect(STATUSES).toContain(surface.status)
    expect(surface.story).toMatch(/^A-\d+[a-z]?$/)
    expect(surface.names.length).toBeGreaterThan(0)
    expect(surface.names.filter((name) => !IDENTIFIER.test(name))).toEqual([])
    expect([...surface.names]).toEqual([...new Set(surface.names)].sort())
  })

  it('never plans one name in two areas (export * would drop it)', () => {
    expect(PLANNED.filter((name, index) => PLANNED.indexOf(name) !== index)).toEqual([])
  })

  it('marks the five areas A-10a builds and the area of each landed lane as built, and every other area as a stub', () => {
    const built = SURFACES.filter((surface) => surface.status === 'built').map((surface) => surface.area)
    expect(built.sort()).toEqual([...BUILT_BY_THIS_STORY, ...BUILT_BY_LATER_LANES].sort())
  })

  it('re-exports every area from index.ts', () => {
    const index = readFileSync(path.join(SDK_DIR, 'index.ts'), 'utf8')
    const reexported = Array.from(index.matchAll(/^export \* from '\.\/([a-z/-]+)'$/gm), (match) => match[1])
    expect(reexported.sort()).toEqual(Object.keys(AREA_MODULES).sort())
  })
})

describe('PRD-261 A-10a — the surface snapshot', () => {
  it('reports an unplanned export and a missing planned name', () => {
    expect(surfaceDrift(['notify', 'sneakyExtra'], ['notify', 'dismissNotice'])).toEqual({
      unplanned: ['sneakyExtra'],
      missing: ['dismissNotice'],
    })
  })

  it.each(SURFACES.map(areaCase))("%s exports exactly its surface file's names", async (area, surface) => {
    const exported = runtimeNames(await loadArea(area))
    expect(surfaceDrift(exported, surface.names)).toEqual({ unplanned: [], missing: [] })
  })

  it("'@automatos/theme-sdk' exports exactly the union of the surface files", async () => {
    const exported = runtimeNames(await import('@automatos/theme-sdk'))
    expect(surfaceDrift(exported, PLANNED)).toEqual({ unplanned: [], missing: [] })
  })

  it('is the same module by its alias and by its path', async () => {
    const byAlias = await import('@automatos/theme-sdk')
    const byPath = await import('../index')
    expect(byAlias.useWorkspace).toBe(byPath.useWorkspace)
    expect(byAlias.notify).toBe(byPath.notify)
  })
})

describe('PRD-261 A-10a — stubbed areas', () => {
  const stubs = SURFACES.filter((surface) => surface.status === 'stub')

  it.each(stubs.map(areaCase))('%s: every name throws "not in this SDK build"', async (area, surface) => {
    const module = await loadArea(area)
    for (const name of surface.names) {
      const stub = module[name] as (...args: unknown[]) => unknown
      expect(typeof stub, name).toBe('function')
      expect(stub.name, name).toBe(name)
      expect(() => stub(), name).toThrow(`Automatos theme SDK: ${name} (${area}) is ${NOT_IN_THIS_BUILD}.`)
    }
  })
})
