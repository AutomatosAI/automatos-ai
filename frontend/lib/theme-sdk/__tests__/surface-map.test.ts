/**
 * PRD-261 A-10c (PA-16) — the SDK surface map, docs/theme-host/sdk-surface-map.md, held to the SDK's
 * surface files (lib/theme-sdk/surface/*.json) only.
 *
 * - Every `area.name` the map gives a module is planned in that area's surface file, and every
 *   surface name appears in the map: no SDK name without a reason, no reason without a name.
 * - Each data hook the Q3 pages read (plan §0) has a row, and so does each shared module a theme
 *   imports (React, the SDK, Play's icons).
 *
 * PRD-261 A-19f2: Play's code left core (A-19f1), so the static scan of Play's folders that held
 * their imports to the map is the theme kit's now (themes/play/test/imports.test.ts: no `@/` import,
 * only the host's shared modules); the map's **Read by** column records where Play read each module
 * while it sat in core.
 */
import { readdirSync, readFileSync } from 'fs'
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

const SDK_NAME = /`([a-z]+(?:\/[a-z-]+)?)\.([A-Za-z_$][\w$]*)`/g
const MODULE_CELL = /^`([^`]+)`$/
const WILDCARD = '/*'

const MAP_VALUES = ['SDK', "a theme's own", 'host-only', 'shared module']
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
/** The shared modules a theme imports (Play's, since the W-A5 port reached the core through the SDK). */
const SHARED_IMPORTS = ['@automatos/theme-sdk', 'lucide-react', 'react']

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

const SURFACES: readonly SurfaceFile[] = readdirSync(SURFACE_DIR)
  .filter((file) => file.endsWith('.json'))
  .map((file) => JSON.parse(readFileSync(path.join(SURFACE_DIR, file), 'utf8')) as SurfaceFile)
const PLANNED = new Map(SURFACES.map((surface) => [surface.area, new Set(surface.names)]))
const TABLES = tablesOf(readFileSync(MAP_FILE, 'utf8'))
const ROWS = mapRowsOf(TABLES)

describe('PRD-261 A-10c — the map file', () => {
  it('has a row per module, once each, with a known Map value', () => {
    expect(ROWS.length).toBeGreaterThan(Q3_DATA_HOOKS.length)
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

  it('has a row for each shared module a theme imports', () => {
    expect(uncovered(SHARED_IMPORTS, ROWS.map((row) => row.module))).toEqual([])
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

describe('PRD-261 A-10c — the helpers', () => {
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
