/**
 * PRD-261 A-12d — the SDK agents-page hello theme the browser lane serves (e2e/browser/fixtures/hello-a12d,
 * specs/pages-agents.spec.ts) is valid as files, and is a theme the way the PRD means it: its entry imports
 * only the shared modules, takes from the SDK only names a built area exports, registers its page on /agents
 * and reads it from the pages/agents hooks; its stylesheet stays under its own class; its test data has the
 * shape the backend answers (AgentResponse rows and the agent-statistics counts).
 */
import { readdirSync, readFileSync } from 'fs'
import path from 'path'
import ts from 'typescript'
import { describe, expect, it } from 'vitest'

import sharedModules from '@/lib/theme-host/shared-modules.json'

interface SurfaceFile {
  readonly status: 'built' | 'stub'
  readonly names: readonly string[]
}

interface AgentRow {
  readonly id?: unknown
  readonly name?: unknown
  readonly status?: unknown
  readonly agent_type?: unknown
}

interface Responses {
  readonly agents: readonly AgentRow[]
  readonly stats: Readonly<Record<string, unknown>>
  readonly iconMappings: Readonly<Record<string, unknown>>
}

const FIXTURE = path.resolve(__dirname, '../../../../e2e/browser/fixtures/hello-a12d')
const SURFACE_DIR = path.resolve(__dirname, '../surface')
const SDK_SPECIFIER = '@automatos/theme-sdk'
const HTML_CLASS = 'hello-a12d'
const ENTRY = readFileSync(path.join(FIXTURE, 'entry.js'), 'utf8')
const STYLESHEET = readFileSync(path.join(FIXTURE, 'theme.css'), 'utf8')
const RESPONSES = JSON.parse(readFileSync(path.join(FIXTURE, 'responses.json'), 'utf8')) as Responses
const IMPORT = /^import\s+(?:\{([^}]*)\}|[\w$]+)\s+from\s+'([^']+)'/gm
/** Each rule's selector list: the text before a `{`, comments removed. */
const SELECTOR = /([^{}]+)\{/g
const COMMENT = /\/\*[\s\S]*?\*\//g
const STATS_COUNTS = ['active_agents', 'inactive_agents', 'total_agents']
/** The system config key useAgents reads its icon mappings from (hooks/use-system-config-api.ts). */
const ICON_MAPPINGS_KEY = 'system_icon_mappings'

function syntaxErrors(source: string, fileName: string): string[] {
  const output = ts.transpileModule(source, {
    reportDiagnostics: true,
    fileName,
    compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022, allowJs: true },
  })
  return (output.diagnostics ?? []).map((diagnostic) => ts.flattenDiagnosticMessageText(diagnostic.messageText, '\n'))
}

function importsOf(source: string): { specifier: string; names: string[] }[] {
  return Array.from(source.matchAll(IMPORT), (match) => ({
    specifier: match[2],
    names: (match[1] ?? '').split(',').map((name) => name.trim()).filter((name) => name.length > 0),
  }))
}

function builtNames(): string[] {
  return readdirSync(SURFACE_DIR)
    .filter((file) => file.endsWith('.json'))
    .map((file) => JSON.parse(readFileSync(path.join(SURFACE_DIR, file), 'utf8')) as SurfaceFile)
    .filter((surface) => surface.status === 'built')
    .flatMap((surface) => surface.names)
}

function selectors(css: string): string[] {
  return Array.from(css.replace(COMMENT, '').matchAll(SELECTOR), (match) => match[1])
    .flatMap((list) => list.split(','))
    .map((selector) => selector.trim())
    .filter((selector) => selector.length > 0)
}

function isAgentRow(row: AgentRow): boolean {
  return Number.isInteger(row.id) && typeof row.name === 'string' && typeof row.status === 'string' && typeof row.agent_type === 'string'
}

describe('PRD-261 A-12d — the SDK agents-page hello theme', () => {
  it('has its entry, manifest, stylesheet, preset, test data and README', () => {
    expect(readdirSync(FIXTURE).sort()).toEqual(['README.md', 'entry.js', 'presets', 'responses.json', 'theme.css', 'theme.json'])
    const manifest = JSON.parse(readFileSync(path.join(FIXTURE, 'theme.json'), 'utf8')) as Record<string, unknown>
    expect(manifest).toMatchObject({ entry: 'entry.js', styleSheet: 'theme.css', preset: 'presets/default.look.json', htmlClass: HTML_CLASS })
    expect(() => JSON.parse(readFileSync(path.join(FIXTURE, 'presets', 'default.look.json'), 'utf8'))).not.toThrow()
  })

  it('parses as an ES module', () => {
    expect(syntaxErrors(ENTRY, 'entry.js')).toEqual([])
  })

  it('imports only modules the host shares, and never sonner, next/*, react-query or an @/ path', () => {
    const shared = sharedModules.modules.map((entry) => entry.specifier)
    const imported = importsOf(ENTRY).map((entry) => entry.specifier)
    expect(imported.filter((specifier) => !shared.includes(specifier))).toEqual([])
    expect(ENTRY).not.toMatch(/['"](sonner|next\/[^'"]*|@tanstack\/[^'"]*|@\/[^'"]*)['"]/)
  })

  it('takes from the SDK only names a built area exports: the pages/agents hooks', () => {
    const fromSdk = importsOf(ENTRY).filter((entry) => entry.specifier === SDK_SPECIFIER).flatMap((entry) => entry.names)
    expect(fromSdk).toEqual(expect.arrayContaining(['defineTheme', 'useAgents', 'useAgentStats']))
    expect(fromSdk.filter((name) => !builtNames().includes(name))).toEqual([])
  })

  it('registers its page on /agents and draws it from both hooks, refreshing through refetch', () => {
    expect(ENTRY).toMatch(/const AGENTS_ROUTE = '\/agents'/)
    expect(ENTRY).toMatch(/host\.registerPage\(AGENTS_ROUTE, HelloAgentsPage\)/)
    expect(ENTRY).toMatch(/useAgents\(\)/)
    expect(ENTRY).toMatch(/useAgentStats\(\)/)
    expect(ENTRY).toMatch(/agents\.refetch\(\)/)
  })

  it('keeps every rule under its own class', () => {
    const all = selectors(STYLESHEET)
    expect(all.length).toBeGreaterThan(0)
    expect(all.filter((selector) => !selector.startsWith(`html.${HTML_CLASS} `))).toEqual([])
    expect(STYLESHEET).not.toMatch(/!important/)
  })
})

describe('PRD-261 A-12d — the test data', () => {
  it('lists agents as the backend answers them, each with a distinct id and name', () => {
    expect(RESPONSES.agents.length).toBeGreaterThan(1)
    expect(RESPONSES.agents.filter((row) => !isAgentRow(row))).toEqual([])
    expect(new Set(RESPONSES.agents.map((row) => row.id)).size).toBe(RESPONSES.agents.length)
    expect(new Set(RESPONSES.agents.map((row) => row.name)).size).toBe(RESPONSES.agents.length)
  })

  it('counts more agents than the roster lists, so the page shows which read a number came from', () => {
    for (const key of STATS_COUNTS) expect(Number.isInteger(RESPONSES.stats[key]), key).toBe(true)
    expect(RESPONSES.stats.total_agents).not.toBe(RESPONSES.agents.length)
    expect(RESPONSES.stats.active_agents).not.toBe(RESPONSES.agents.filter((row) => row.status === 'active').length)
  })

  it("answers the icon-mapping read with no mappings, so the roster's query key never changes under the spec", () => {
    expect(RESPONSES.iconMappings.config_key).toBe(ICON_MAPPINGS_KEY)
    expect(RESPONSES.iconMappings.config_value).toEqual({})
  })
})
