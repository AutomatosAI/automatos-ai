/**
 * PRD-261 A-12g — the playbooks-data hello theme the browser lane serves (e2e/browser/fixtures/hello-a12g,
 * specs/pages-playbooks.spec.ts) is valid as files, and is a theme the way the PRD means it: its entry imports
 * only the shared modules, takes from the SDK only names a built area exports (a stub would throw in the lane),
 * reads its page's data through the pages/playbooks area, registers its page on /playbooks of the host's page
 * list, and never imports sonner, an `@/` path or `next/*`. The test data the spec answers with has the shapes
 * the entry reads, and template ids the SDK accepts.
 */
import { readdirSync, readFileSync } from 'fs'
import path from 'path'
import ts from 'typescript'
import { describe, expect, it } from 'vitest'

import { THEME_PAGE_ROUTES } from '@/lib/theme-host/constants'
import sharedModules from '@/lib/theme-host/shared-modules.json'

interface SurfaceFile {
  readonly area: string
  readonly status: 'built' | 'stub'
  readonly names: readonly string[]
}

interface PlaybookRow {
  readonly id: number
  readonly template_id: string
  readonly name: string
  readonly steps: readonly unknown[]
}

interface Responses {
  readonly playbookList: { readonly items: readonly PlaybookRow[]; readonly total: number }
  readonly playbookDetail: PlaybookRow
  readonly runs: { readonly items: readonly { readonly execution_id: string; readonly status: string }[]; readonly recipe_id: string }
  readonly runStarted: { readonly recipe_execution_id: string; readonly recipe_id: string }
}

const FIXTURE = path.resolve(__dirname, '../../../../e2e/browser/fixtures/hello-a12g')
const SURFACE_DIR = path.resolve(__dirname, '../surface')
const SDK_SPECIFIER = '@automatos/theme-sdk'
const AREA = 'pages/playbooks'
const ENTRY = readFileSync(path.join(FIXTURE, 'entry.js'), 'utf8')
const RESPONSES = JSON.parse(readFileSync(path.join(FIXTURE, 'responses.json'), 'utf8')) as Responses
const IMPORT = /^import\s+(?:\{([^}]*)\}|[\w$]+)\s+from\s+'([^']+)'/gm
const REGISTERED_PAGE = /host\.registerPage\('([^']+)'/g
/** The SDK's own template id check (lib/theme-sdk/pages/playbooks.ts). */
const TEMPLATE_ID = /^[A-Za-z0-9][A-Za-z0-9_.-]{0,199}$/
const PLAYBOOKS_ROUTES = ['/playbooks']
const AREA_HOOKS_READ = ['useExecutePlaybook', 'usePlaybookExecutions', 'useWorkflowPlaybook', 'useWorkflowPlaybooks']

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

function surfaces(): SurfaceFile[] {
  return readdirSync(SURFACE_DIR)
    .filter((file) => file.endsWith('.json'))
    .map((file) => JSON.parse(readFileSync(path.join(SURFACE_DIR, file), 'utf8')) as SurfaceFile)
}

function sdkNames(): string[] {
  return importsOf(ENTRY).filter((entry) => entry.specifier === SDK_SPECIFIER).flatMap((entry) => entry.names)
}

describe('PRD-261 A-12g — the playbooks-data hello theme', () => {
  it('has its entry, manifest, stylesheet, preset, test data and README', () => {
    expect(readdirSync(FIXTURE).sort()).toEqual(['README.md', 'entry.js', 'presets', 'responses.json', 'theme.css', 'theme.json'])
    const manifest = JSON.parse(readFileSync(path.join(FIXTURE, 'theme.json'), 'utf8')) as Record<string, unknown>
    expect(manifest).toMatchObject({ entry: 'entry.js', styleSheet: 'theme.css', preset: 'presets/default.look.json', htmlClass: 'hello-a12g' })
    expect(() => JSON.parse(readFileSync(path.join(FIXTURE, 'presets', 'default.look.json'), 'utf8'))).not.toThrow()
  })

  it('parses as an ES module', () => {
    expect(syntaxErrors(ENTRY, 'entry.js')).toEqual([])
  })

  it('imports only modules the host shares', () => {
    const shared = sharedModules.modules.map((entry) => entry.specifier)
    const imported = importsOf(ENTRY).map((entry) => entry.specifier)
    expect(imported.length).toBeGreaterThan(0)
    expect(imported.filter((specifier) => !shared.includes(specifier))).toEqual([])
    expect(ENTRY).not.toMatch(/['"](sonner|next\/[^'"]*|@\/[^'"]*)['"]/)
  })

  it('takes from the SDK only names a built area exports, and reads its data through pages/playbooks', () => {
    const built = surfaces().filter((surface) => surface.status === 'built').flatMap((surface) => surface.names)
    const area = surfaces().find((surface) => surface.area === AREA)
    expect(area?.status).toBe('built')
    expect(sdkNames().filter((name) => !built.includes(name))).toEqual([])
    for (const hook of AREA_HOOKS_READ) {
      expect(area?.names).toContain(hook)
      expect(sdkNames()).toContain(hook)
      expect(ENTRY).toMatch(new RegExp(`\\b${hook}\\(`))
    }
  })

  it("registers its page on /playbooks of the host's page list", () => {
    const registered = Array.from(ENTRY.matchAll(REGISTERED_PAGE), (match) => match[1]).sort()
    expect(registered).toEqual(PLAYBOOKS_ROUTES)
    expect(registered.filter((route) => !THEME_PAGE_ROUTES.includes(route))).toEqual([])
  })

  it('answers with test data the page can read, under template ids the SDK accepts', () => {
    const { playbookList, playbookDetail, runs, runStarted } = RESPONSES
    expect(playbookList.items.length).toBeGreaterThan(1)
    expect(playbookList.total).toBe(playbookList.items.length)
    for (const playbook of [...playbookList.items, playbookDetail]) {
      expect(playbook.template_id).toMatch(TEMPLATE_ID)
      expect(playbook.name.length).toBeGreaterThan(0)
    }
    expect(playbookList.items.map((playbook) => playbook.template_id)).toContain(playbookDetail.template_id)
    expect(playbookDetail.steps.length).toBeGreaterThan(0)
    expect(runs.items.length).toBeGreaterThan(0)
    expect(runs.recipe_id).toBe(playbookDetail.template_id)
    expect(runStarted.recipe_id).toBe(playbookDetail.template_id)
    expect(runs.items.map((run) => run.execution_id)).not.toContain(runStarted.recipe_execution_id)
  })
})
