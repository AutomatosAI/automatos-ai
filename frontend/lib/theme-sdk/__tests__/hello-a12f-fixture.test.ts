/**
 * PRD-261 A-12f — the knowledge-data hello theme the browser lane serves
 * (e2e/browser/fixtures/hello-a12f, specs/pages-documents.spec.ts) is valid as files, and is a
 * theme the way the PRD means it: its entry imports only the shared modules, takes from the SDK
 * only names a built area exports (a stub would throw in the lane), registers its page on
 * /documents, reads the pages/documents hooks there, and never imports sonner, react-query, an
 * `@/` path or `next/*`. Its test data is JSON with every answer the spec serves.
 */
import { readdirSync, readFileSync } from 'fs'
import path from 'path'
import ts from 'typescript'
import { describe, expect, it } from 'vitest'

import sharedModules from '@/lib/theme-host/shared-modules.json'
import documentsSurface from '../surface/pages-documents.json'

interface SurfaceFile {
  readonly status: 'built' | 'stub'
  readonly names: readonly string[]
}

const FIXTURE = path.resolve(__dirname, '../../../../e2e/browser/fixtures/hello-a12f')
const SURFACE_DIR = path.resolve(__dirname, '../surface')
const SDK_SPECIFIER = '@automatos/theme-sdk'
const ENTRY = readFileSync(path.join(FIXTURE, 'entry.js'), 'utf8')
const IMPORT = /^import\s+(?:\{([^}]*)\}|[\w$]+)\s+from\s+'([^']+)'/gm
const FIXTURE_FILES = ['README.md', 'entry.js', 'presets', 'responses.json', 'theme.css', 'theme.json']
const READ_IN_THE_SPEC = ['useCloudConnections', 'useDatabaseKnowledge', 'useDeleteDocument', 'useDocumentStats',
  'useDocumentTeamCounts', 'useDocuments', 'useTeams']
const RESPONSE_KEYS = ['connections', 'databaseSources', 'documents', 'stats', 'teamCounts', 'teams', 'templates']

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

function readJson(file: string): Record<string, unknown> {
  return JSON.parse(readFileSync(path.join(FIXTURE, file), 'utf8')) as Record<string, unknown>
}

describe('PRD-261 A-12f — the knowledge-data hello theme', () => {
  it('has its entry, manifest, stylesheet, preset, README and test data', () => {
    expect(readdirSync(FIXTURE).sort()).toEqual(FIXTURE_FILES)
    expect(readJson('theme.json')).toMatchObject({
      entry: 'entry.js',
      styleSheet: 'theme.css',
      preset: 'presets/default.look.json',
      htmlClass: 'hello-a12f',
    })
    expect(() => readJson(path.join('presets', 'default.look.json'))).not.toThrow()
  })

  it('has a test answer for every endpoint the spec serves', () => {
    const responses = readJson('responses.json')
    expect(Object.keys(responses).filter((key) => !key.startsWith('_')).sort()).toEqual(RESPONSE_KEYS)
    const documents = responses.documents
    expect(Array.isArray(documents) ? documents.length : 0).toBeGreaterThan(1)
  })

  it('parses as an ES module', () => {
    expect(syntaxErrors(ENTRY, 'entry.js')).toEqual([])
  })

  it('imports only modules the host shares, and never sonner, react-query, next or an @/ path', () => {
    const shared = sharedModules.modules.map((entry) => entry.specifier)
    const imported = importsOf(ENTRY).map((entry) => entry.specifier)
    expect(imported.length).toBeGreaterThan(0)
    expect(imported.filter((specifier) => !shared.includes(specifier))).toEqual([])
    expect(ENTRY).not.toMatch(/['"](sonner|@tanstack\/[^'"]*|next\/[^'"]*|@\/[^'"]*)['"]/)
  })

  it('takes from the SDK only built names, the pages/documents hooks among them, and pages /documents', () => {
    const fromSdk = importsOf(ENTRY).filter((entry) => entry.specifier === SDK_SPECIFIER).flatMap((entry) => entry.names)
    expect(fromSdk.filter((name) => !builtNames().includes(name))).toEqual([])
    expect(READ_IN_THE_SPEC.filter((name) => !fromSdk.includes(name))).toEqual([])
    expect(READ_IN_THE_SPEC.filter((name) => !documentsSurface.names.includes(name))).toEqual([])
    expect(ENTRY).toMatch(/registerPage\('\/documents', /)
  })
})
