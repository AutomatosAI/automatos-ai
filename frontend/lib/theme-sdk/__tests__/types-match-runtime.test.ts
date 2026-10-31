/**
 * PRD-261 A-10b — the SDK's types package (frontend/packages/theme-sdk) held to the SDK's runtime
 * (frontend/lib/theme-sdk), so the kit vendors exactly what the host provides.
 *
 * - Every runtime name of an area is declared as a value in that area's .d.ts, and the .d.ts declares
 *   no value the runtime lacks (a theme can import nothing that is undefined at runtime).
 * - Every type a runtime area exports is declared in its .d.ts.
 * - A stub area declares each name as NotInThisBuild; a built area declares none that way, so the lane
 *   that fills an area's runtime replaces its declarations in the same change.
 * - index.d.ts re-exports exactly the areas; the declarations import only react and their own files.
 * - package.json: the SDK's name, version = sdkVersion, an @types/react 19 peer, types only.
 * - tsc (CI's baselined gate) holds the host's module to the declared types: RUNTIME_AS_DECLARED.
 */
import { existsSync, readdirSync, readFileSync } from 'fs'
import path from 'path'
import { describe, expect, it } from 'vitest'

import type * as Declared from '../../../packages/theme-sdk/index'
import * as Runtime from '../index'

interface SurfaceFile {
  readonly area: string
  readonly status: 'built' | 'stub'
  readonly names: readonly string[]
}

interface TypesPackage {
  readonly name?: unknown
  readonly version?: unknown
  readonly private?: unknown
  readonly types?: unknown
  readonly main?: unknown
  readonly files?: unknown
  readonly dependencies?: unknown
  readonly peerDependencies?: Readonly<Record<string, unknown>>
}

type ModuleLoader = () => Promise<unknown>

/**
 * The host's SDK module, typed as the types package declares it. If any runtime export does not fit its
 * declaration, this line is a tsc error and the baselined tsc gate fails.
 */
const RUNTIME_AS_DECLARED: typeof Declared = Runtime

const FRONTEND = path.resolve(__dirname, '../../..')
const SDK_DIR = path.join(FRONTEND, 'lib/theme-sdk')
const SURFACE_DIR = path.join(SDK_DIR, 'surface')
const PACKAGE_DIR = path.join(FRONTEND, 'packages/theme-sdk')
const AREAS_DIR = path.join(PACKAGE_DIR, 'areas')
const SHARED_MODULES = path.join(FRONTEND, 'lib/theme-host/shared-modules.json')
const REGISTRY_TYPES = path.join(FRONTEND, 'lib/theme-host/registry-types.ts')
const SKIPPED = new Set(['default', '__esModule'])
const SDK_NAME = '@automatos/theme-sdk'
const REACT_TYPES = '@types/react'
const REACT_TYPES_MAJOR = /^\^19\./
const STUB_TYPE = 'NotInThisBuild'

const DECLARED_VALUE = /^export declare (?:const|function|class|let|var) ([A-Za-z_$][\w$]*)/gm
const DECLARED_TYPE = /^export (?:declare )?(?:type|interface) ([A-Za-z_$][\w$]*)/gm
const DECLARED_STUB = new RegExp(`^export declare const ([A-Za-z_$][\\w$]*): ${STUB_TYPE}$`, 'gm')
const RUNTIME_TYPE = /^export (?:type|interface) ([A-Za-z_$][\w$]*)/gm
const RUNTIME_TYPE_LIST = /^export type \{([^}]*)\}/gm
const MODULE_SPECIFIER = /^(?:import|export)\b[^';]*?from '([^']+)'/gm
const INDEX_AREA = /^export \* from '\.\/areas\/([a-z-]+)'$/gm
const QUOTED = /'([^']+)'/g
/** `export type SlotId =` and its continuation lines (each starting with `|`). */
const SLOT_ID_TYPE = /^export type SlotId =(.*(?:\n\s+\|.*)*)/m

/** Every area and its runtime module, written out so the bundler sees each import. */
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
  storage: () => import('../storage'),
  voice: () => import('../voice'),
  proactive: () => import('../proactive'),
}

const SURFACES: readonly SurfaceFile[] = readdirSync(SURFACE_DIR)
  .filter((file) => file.endsWith('.json'))
  .sort()
  .map((file) => JSON.parse(readFileSync(path.join(SURFACE_DIR, file), 'utf8')) as SurfaceFile)

function read(file: string): string {
  return readFileSync(file, 'utf8')
}

function sortedUnique(names: readonly string[]): string[] {
  return Array.from(new Set(names)).sort()
}

function captured(text: string, pattern: RegExp): string[] {
  return sortedUnique(Array.from(text.matchAll(pattern), (match) => match[1]))
}

/** The area's file name in the types package: pages/agents → pages-agents. */
function flatName(area: string): string {
  return area.replace('/', '-')
}

function declarationOf(area: string): string {
  return read(path.join(AREAS_DIR, `${flatName(area)}.d.ts`))
}

/** The runtime source of an area: lib/theme-sdk/<area>.ts, or .tsx for one that renders. */
function runtimeSourceOf(area: string): string {
  const base = path.join(SDK_DIR, area)
  return read(existsSync(`${base}.ts`) ? `${base}.ts` : `${base}.tsx`)
}

/** The type names a runtime area exports: its own declarations and its `export type { … }` lists. */
function runtimeTypeNames(source: string): string[] {
  const listed = Array.from(source.matchAll(RUNTIME_TYPE_LIST), (match) => match[1])
    .flatMap((list) => list.split(','))
    .map((entry) => entry.trim().split(/\s+as\s+/).pop() ?? '')
    .filter((name) => name.length > 0)
  return sortedUnique([...captured(source, RUNTIME_TYPE), ...listed])
}

function runtimeNames(module: unknown): string[] {
  return Object.keys(module as Record<string, unknown>).filter((name) => !SKIPPED.has(name)).sort()
}

/** What the runtime has that the declarations lack, and what the declarations have that the runtime lacks. */
function valueDrift(runtime: readonly string[], declared: readonly string[]): { undeclared: string[]; phantom: string[] } {
  return {
    undeclared: runtime.filter((name) => !declared.includes(name)),
    phantom: declared.filter((name) => !runtime.includes(name)),
  }
}

/** Every .d.ts under the types package, relative to it. */
function declarationFiles(dir: string = PACKAGE_DIR): string[] {
  return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const full = path.join(dir, entry.name)
    if (entry.isDirectory()) return declarationFiles(full)
    return entry.name.endsWith('.d.ts') ? [path.relative(PACKAGE_DIR, full)] : []
  })
}

function slotIds(text: string): string[] {
  const declaration = SLOT_ID_TYPE.exec(text)
  return declaration ? captured(declaration[1], QUOTED) : []
}

function areaCase(surface: SurfaceFile): [string, SurfaceFile] {
  return [surface.area, surface]
}

describe('PRD-261 A-10b — each area is declared as it runs', () => {
  it('reports an undeclared runtime name and a declared name the runtime lacks', () => {
    expect(valueDrift(['notify', 'dismissNotice'], ['notify', 'ghost'])).toEqual({
      undeclared: ['dismissNotice'],
      phantom: ['ghost'],
    })
  })

  it('has one declaration file per surface area, and no other', () => {
    const files = readdirSync(AREAS_DIR).filter((file) => file.endsWith('.d.ts')).sort()
    expect(files).toEqual(SURFACES.map((surface) => `${flatName(surface.area)}.d.ts`).sort())
  })

  it.each(SURFACES.map(areaCase))('%s declares exactly its runtime names', async (area) => {
    const runtime = runtimeNames(await AREA_MODULES[area]())
    expect(valueDrift(runtime, captured(declarationOf(area), DECLARED_VALUE))).toEqual({ undeclared: [], phantom: [] })
  })

  it.each(SURFACES.map(areaCase))('%s declares every type its runtime exports', (area) => {
    const declared = captured(declarationOf(area), DECLARED_TYPE)
    expect(runtimeTypeNames(runtimeSourceOf(area)).filter((name) => !declared.includes(name))).toEqual([])
  })

  it.each(SURFACES.map(areaCase))('%s declares stubs exactly while its runtime is a stub', (area, surface) => {
    const stubs = captured(declarationOf(area), DECLARED_STUB)
    expect(stubs).toEqual(surface.status === 'stub' ? sortedUnique(surface.names) : [])
  })

  it('declares the host slot ids the registry has', () => {
    const declared = slotIds(read(path.join(PACKAGE_DIR, 'internal/host.d.ts')))
    expect(declared.length).toBeGreaterThan(0)
    expect(declared).toEqual(slotIds(read(REGISTRY_TYPES)))
  })

  it("holds the host's module to the declared types (tsc proves the fit; this proves it is the same module)", () => {
    expect(RUNTIME_AS_DECLARED).toBe(Runtime)
  })
})

describe('PRD-261 A-10b — the index and the imports', () => {
  it('re-exports every area from index.d.ts, once each', () => {
    const reexported = Array.from(read(path.join(PACKAGE_DIR, 'index.d.ts')).matchAll(INDEX_AREA), (match) => match[1])
    expect(reexported.sort()).toEqual(SURFACES.map((surface) => flatName(surface.area)).sort())
  })

  it.each(declarationFiles().map((file) => [file]))('%s imports only react and its own package', (file) => {
    const specifiers = captured(read(path.join(PACKAGE_DIR, file)), MODULE_SPECIFIER)
    const foreign = specifiers.filter((specifier) => specifier !== 'react' && !specifier.startsWith('.'))
    expect(foreign).toEqual([])
    for (const specifier of specifiers.filter((name) => name.startsWith('.'))) {
      const target = path.resolve(PACKAGE_DIR, path.dirname(file), `${specifier}.d.ts`)
      expect(target.startsWith(PACKAGE_DIR + path.sep), specifier).toBe(true)
      expect(existsSync(target), specifier).toBe(true)
    }
  })
})

describe('PRD-261 A-10b — package.json', () => {
  const pkg = JSON.parse(read(path.join(PACKAGE_DIR, 'package.json'))) as TypesPackage
  const sdkVersion = (JSON.parse(read(SHARED_MODULES)) as { sdkVersion: string }).sdkVersion

  it("is the SDK's types, at the SDK's version", () => {
    expect(pkg.name).toBe(SDK_NAME)
    expect(pkg.version).toBe(sdkVersion)
    expect(pkg.version).toBe(Runtime.SDK_VERSION)
  })

  it('is types only and never published to a registry (D2-b: npm pack artefact, vendored by the kit)', () => {
    expect(pkg.private).toBe(true)
    expect(pkg.types).toBe('./index.d.ts')
    expect(pkg.main).toBeUndefined()
    expect(pkg.dependencies).toBeUndefined()
  })

  it('asks for React 19 types as a peer', () => {
    expect(String(pkg.peerDependencies?.[REACT_TYPES] ?? '')).toMatch(REACT_TYPES_MAJOR)
  })

  it('packs every declaration file and the README', () => {
    const files = Array.isArray(pkg.files) ? (pkg.files as unknown[]).map(String) : []
    for (const entry of files) expect(existsSync(path.join(PACKAGE_DIR, entry)), entry).toBe(true)
    const unpacked = declarationFiles().filter((file) => !files.some((entry) => file === entry || file.startsWith(entry)))
    expect(unpacked).toEqual([])
    expect(files).toContain('README.md')
  })
})
