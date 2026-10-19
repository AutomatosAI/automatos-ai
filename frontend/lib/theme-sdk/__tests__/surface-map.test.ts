/**
 * PRD-261 A-10c (PA-16) — the SDK surface map, docs/theme-host/sdk-surface-map.md, held to the code.
 *
 * - Every `area.name` the map gives a module is planned in that area's surface file, and every
 *   surface name appears in the map: no SDK name without a reason, no reason without a name.
 * - A static scan of Play's folders (components/{play,looks,mascot}, lib/{play,looks,mascot}; their
 *   runtime files, not their tests) finds no `@/` import outside those folders, and no package
 *   import, that has no row. A new import there is planned in the map first.
 * - Each data hook the Q3 pages read (plan §0) has a row.
 */
import { readdirSync, readFileSync, statSync } from 'fs'
import path from 'path'
import { describe, expect, it } from 'vitest'

interface SurfaceFile {
  readonly area: string
  readonly names: readonly string[]
}

interface MapRow {
  readonly module: string
  readonly map: string
  readonly names: readonly SdkName[]
  readonly why: string
}

interface SdkName {
  readonly area: string
  readonly name: string
}

type Table = readonly Readonly<Record<string, string>>[]

const FRONTEND = path.resolve(__dirname, '../../..')
const MAP_FILE = path.resolve(FRONTEND, '../docs/theme-host/sdk-surface-map.md')
const SURFACE_DIR = path.join(FRONTEND, 'lib/theme-sdk/surface')

const PLAY_FOLDERS = ['components/play', 'components/looks', 'components/mascot', 'lib/play', 'lib/looks', 'lib/mascot']
const PLAY_OWN = /^@\/(components\/(play|looks|mascot)|lib\/(play|looks|mascot))(\/|$)/
const SOURCE_FILE = /\.(tsx?|jsx?)$/
const TEST_FILE = /\.test\.[jt]sx?$/
const TEST_DIR = '__tests__'
/**
 * ` from '…'`, `import('…')` and a bare `import '…'`, in single or double quotes. The specifier has no
 * whitespace, and `from` follows whitespace, so a key such as `'--gradient-from': …` is not an import.
 */
const IMPORT_SPECIFIER = /(?:\sfrom\s+|\bimport\s*\(\s*|^\s*import\s+)(['"])([@.\w][^'"\s]*)\1/gm
const SDK_NAME = /`([a-z]+(?:\/[a-z-]+)?)\.([A-Za-z_$][\w$]*)`/g
const MODULE_CELL = /^`([^`]+)`$/
const WILDCARD = '/*'

const MAP_VALUES = ['SDK', 'copied into Play', 'host-only', 'shared module']
const NO_NAMES = '—'
/** The data hooks the Q3 pages read (plan §0, and the /missions/[id] hooks behind them). */
const Q3_DATA_HOOKS = [
  '@/hooks/use-agent-api',
  '@/hooks/use-page-api',
  '@/hooks/use-missions-api',
  '@/hooks/use-assignments-api',
  '@/hooks/use-playbook-api',
  '@/hooks/use-document-api',
  '@/hooks/use-teams',
  '@/hooks/use-database-knowledge',
  '@/hooks/use-cloud-storage',
  '@/hooks/use-activity-api',
  '@/hooks/use-approval-grants',
  '@/hooks/use-board-event-stream',
  '@/hooks/use-needs-you',
  '@/hooks/use-watches-api',
  '@/components/workspace-provider',
]
/** Imports the scan must see, so a broken pattern can never pass by finding nothing. */
const KNOWN_IMPORTS = ['@/types', '@/lib/ticket-links', '@/hooks/use-deliverables-api', 'next/link', 'react']

function cellsOf(line: string): string[] {
  return line.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map((cell) => cell.trim())
}

function isDivider(line: string): boolean {
  return /^\|(\s*:?-+:?\s*\|)+$/.test(line.trim())
}

/** Every Markdown table in `text`, each row keyed by its header's cells. */
function tablesOf(text: string): Table[] {
  const lines = text.split('\n')
  const tables: Table[] = []
  for (let index = 0; index + 1 < lines.length; index += 1) {
    if (!lines[index].trim().startsWith('|') || !isDivider(lines[index + 1])) continue
    const header = cellsOf(lines[index])
    const rows: Record<string, string>[] = []
    let next = index + 2
    for (; next < lines.length && lines[next].trim().startsWith('|'); next += 1) {
      const cells = cellsOf(lines[next])
      rows.push(Object.fromEntries(header.map((key, column) => [key, cells[column] ?? ''])))
    }
    tables.push(rows)
    index = next - 1
  }
  return tables
}

/** The `area.name` references in one cell. */
function sdkNamesOf(cell: string): SdkName[] {
  return Array.from(cell.matchAll(SDK_NAME), (match) => ({ area: match[1], name: match[2] }))
}

function mapRowsOf(tables: readonly Table[]): MapRow[] {
  return tables
    .filter((table) => table.length > 0 && 'Module' in table[0])
    .flatMap((table) => table.map((row) => ({
      module: MODULE_CELL.exec(row.Module)?.[1] ?? row.Module,
      map: row.Map,
      names: sdkNamesOf(row['SDK names'] ?? ''),
      why: row.Why ?? '',
    })))
}

function allSdkNamesOf(tables: readonly Table[]): SdkName[] {
  return tables.flatMap((table) => table.flatMap((row) => sdkNamesOf(row['SDK names'] ?? '')))
}

/** Whether a row's module (exact, or a folder ending in `/*`) covers `specifier`. */
function covers(module: string, specifier: string): boolean {
  if (!module.endsWith(WILDCARD)) return module === specifier
  return specifier.startsWith(module.slice(0, -1))
}

/** The specifiers no row covers. */
function uncovered(specifiers: readonly string[], modules: readonly string[]): string[] {
  return specifiers.filter((specifier) => !modules.some((module) => covers(module, specifier)))
}

function runtimeFilesUnder(dir: string): string[] {
  return readdirSync(dir).flatMap((entry) => {
    const full = path.join(dir, entry)
    if (statSync(full).isDirectory()) return entry === TEST_DIR ? [] : runtimeFilesUnder(full)
    return SOURCE_FILE.test(entry) && !TEST_FILE.test(entry) ? [full] : []
  })
}

/** The `@/` imports outside Play's folders, and the package imports, of one source text. */
function externalImportsOf(source: string): string[] {
  const specifiers = Array.from(source.matchAll(IMPORT_SPECIFIER), (match) => match[2])
  return specifiers.filter((specifier) => (specifier.startsWith('@/') ? !PLAY_OWN.test(specifier) : !specifier.startsWith('.')))
}

function scanPlayFolders(): string[] {
  const files = PLAY_FOLDERS.flatMap((folder) => runtimeFilesUnder(path.join(FRONTEND, folder)))
  const found = files.flatMap((file) => externalImportsOf(readFileSync(file, 'utf8')))
  return Array.from(new Set(found)).sort()
}

const SURFACES: readonly SurfaceFile[] = readdirSync(SURFACE_DIR)
  .filter((file) => file.endsWith('.json'))
  .map((file) => JSON.parse(readFileSync(path.join(SURFACE_DIR, file), 'utf8')) as SurfaceFile)
const PLANNED = new Map(SURFACES.map((surface) => [surface.area, new Set(surface.names)]))
const TABLES = tablesOf(readFileSync(MAP_FILE, 'utf8'))
const ROWS = mapRowsOf(TABLES)
const SCANNED = scanPlayFolders()

describe('PRD-261 A-10c — the map file', () => {
  it('has a row per module, once each, with a known Map value', () => {
    expect(ROWS.length).toBeGreaterThan(KNOWN_IMPORTS.length)
    const modules = ROWS.map((row) => row.module)
    expect(modules.filter((module, index) => modules.indexOf(module) !== index)).toEqual([])
    expect(ROWS.filter((row) => !MAP_VALUES.includes(row.map)).map((row) => row.module)).toEqual([])
    expect(modules.filter((module) => !module.startsWith('@/') && module.includes('*'))).toEqual([])
  })

  it('names an SDK export for every SDK row, and none for any other row, which says why instead', () => {
    expect(ROWS.filter((row) => row.map === 'SDK' && row.names.length === 0).map((row) => row.module)).toEqual([])
    const others = ROWS.filter((row) => row.map !== 'SDK')
    expect(others.filter((row) => row.names.length > 0).map((row) => row.module)).toEqual([])
    expect(others.filter((row) => row.why.length === 0).map((row) => row.module)).toEqual([])
  })

  it('has a row for each data hook the Q3 pages read', () => {
    expect(uncovered(Q3_DATA_HOOKS, ROWS.map((row) => row.module))).toEqual([])
  })
})

describe('PRD-261 A-10c — the map and the surface files', () => {
  it("plans every SDK name the map gives in that area's surface file", () => {
    const unplanned = allSdkNamesOf(TABLES).filter(({ area, name }) => !PLANNED.get(area)?.has(name))
    expect(unplanned.map(({ area, name }) => `${area}.${name}`)).toEqual([])
  })

  it('gives every planned surface name a place in the map', () => {
    const mapped = new Set(allSdkNamesOf(TABLES).map(({ area, name }) => `${area}.${name}`))
    const unmapped = SURFACES.flatMap((surface) => surface.names.map((name) => `${surface.area}.${name}`))
      .filter((qualified) => !mapped.has(qualified))
    expect(unmapped).toEqual([])
  })
})

describe("PRD-261 A-10c — the scan of Play's folders", () => {
  it('sees the imports it must see', () => {
    expect(KNOWN_IMPORTS.filter((specifier) => !SCANNED.includes(specifier))).toEqual([])
    expect(SCANNED.filter((specifier) => PLAY_OWN.test(specifier))).toEqual([])
  })

  it('finds no core or package import the map has no row for', () => {
    expect(uncovered(SCANNED, ROWS.map((row) => row.module))).toEqual([])
  })
})

describe('PRD-261 A-10c — the helpers', () => {
  it('reads from, import() and bare imports, keeps core and package ones, drops relative and Play-own ones', () => {
    const source = [
      "import { a } from '@/lib/ticket-links'",
      'import type { B } from "@/types"',
      "export { c } from '@/hooks/use-online'",
      "const lazy = () => import('@/components/shared/markdown-view')",
      "import '@/app/side-effect.css'",
      "import Link from 'next/link'",
      "import { d } from './local'",
      "import { e } from '@/lib/play/shelf'",
      "import { f } from '@/components/looks/look-provider'",
      "const tokens = { '--background-gradient-from':'x', '--to-from': triplet(bg) }",
    ].join('\n')
    expect(externalImportsOf(source)).toEqual([
      '@/lib/ticket-links', '@/types', '@/hooks/use-online', '@/components/shared/markdown-view',
      '@/app/side-effect.css', 'next/link',
    ])
  })

  it('reports an import no row covers, and lets a folder row cover what is under it', () => {
    const modules = ['@/types', '@/lib/look-kit/*']
    expect(uncovered(['@/types', '@/lib/look-kit/css', '@/lib/look-kits', '@/lib/sneaky'], modules))
      .toEqual(['@/lib/look-kits', '@/lib/sneaky'])
  })

  it('reads tables by header and area.name references in a cell', () => {
    const tables = tablesOf('text\n\n| Module | SDK names |\n|---|---|\n| `@/x` | `chat.useA`, `pages/agents.useB` |\n\nafter')
    expect(tables).toEqual([[{ Module: '`@/x`', 'SDK names': '`chat.useA`, `pages/agents.useB`' }]])
    expect(sdkNamesOf(tables[0][0]['SDK names'])).toEqual([
      { area: 'chat', name: 'useA' },
      { area: 'pages/agents', name: 'useB' },
    ])
    expect(sdkNamesOf(NO_NAMES)).toEqual([])
  })
})
