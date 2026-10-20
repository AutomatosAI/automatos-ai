/**
 * PRD-261 A-12h — the Command Center data hello theme the browser lane serves
 * (e2e/browser/fixtures/hello-a12h, specs/pages-command-center.spec.ts) is valid as files, and is a
 * theme the way the PRD means it: its entry imports only the shared modules, takes from the SDK only
 * names a built area exports (a stub would throw in the lane), calls every pages/command-center
 * hook, registers its page on /command-center, and never imports sonner, an `@/` path or `next/*`.
 */
import { readdirSync, readFileSync } from 'fs'
import path from 'path'
import ts from 'typescript'
import { describe, expect, it } from 'vitest'

import sharedModules from '@/lib/theme-host/shared-modules.json'

interface SurfaceFile {
  readonly area: string
  readonly status: 'built' | 'stub'
  readonly names: readonly string[]
}

const FIXTURE = path.resolve(__dirname, '../../../../e2e/browser/fixtures/hello-a12h')
const SURFACE_DIR = path.resolve(__dirname, '../surface')
const SDK_SPECIFIER = '@automatos/theme-sdk'
const AREA = 'pages/command-center'
const ROUTE = '/command-center'
const ENTRY = readFileSync(path.join(FIXTURE, 'entry.js'), 'utf8')
const IMPORT = /^import\s+(?:\{([^}]*)\}|[\w$]+)\s+from\s+'([^']+)'/gm

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

const SURFACES: readonly SurfaceFile[] = readdirSync(SURFACE_DIR)
  .filter((file) => file.endsWith('.json'))
  .map((file) => JSON.parse(readFileSync(path.join(SURFACE_DIR, file), 'utf8')) as SurfaceFile)

describe('PRD-261 A-12h — the Command Center data hello theme', () => {
  it('has its entry, manifest, stylesheet, preset and README', () => {
    expect(readdirSync(FIXTURE).sort()).toEqual(['README.md', 'entry.js', 'presets', 'theme.css', 'theme.json'])
    const manifest = JSON.parse(readFileSync(path.join(FIXTURE, 'theme.json'), 'utf8')) as Record<string, unknown>
    expect(manifest).toMatchObject({ entry: 'entry.js', styleSheet: 'theme.css', preset: 'presets/default.look.json', htmlClass: 'hello-a12h' })
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

  it('takes from the SDK only names a built area exports, and calls every Command Center hook', () => {
    const fromSdk = importsOf(ENTRY).filter((entry) => entry.specifier === SDK_SPECIFIER).flatMap((entry) => entry.names)
    const built = SURFACES.filter((surface) => surface.status === 'built').flatMap((surface) => surface.names)
    const area = SURFACES.find((surface) => surface.area === AREA)
    expect(area?.status).toBe('built')
    expect(fromSdk.filter((name) => !built.includes(name))).toEqual([])
    for (const name of area?.names ?? []) {
      expect(fromSdk, name).toContain(name)
      expect(ENTRY, name).toMatch(new RegExp(`\\b${name}\\(`))
    }
  })

  it('registers its page on the Command Center route', () => {
    expect(ENTRY).toMatch(/host\.registerPage\(COMMAND_CENTER_ROUTE, /)
    expect(ENTRY).toContain(`const COMMAND_CENTER_ROUTE = '${ROUTE}'`)
  })
})
