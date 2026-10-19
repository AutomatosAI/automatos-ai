/**
 * PRD-261 A-10a — the SDK-context hello theme the browser lane serves
 * (e2e/browser/fixtures/hello-a10a, specs/sdk-context.spec.ts) is valid as files, and is a theme
 * the way the PRD means it: its entry imports only the shared modules, takes from the SDK only
 * names a built area exports (a stub would throw in the lane), calls useWorkspace(), and never
 * imports sonner, an `@/` path or `next/*`.
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

const FIXTURE = path.resolve(__dirname, '../../../../e2e/browser/fixtures/hello-a10a')
const SURFACE_DIR = path.resolve(__dirname, '../surface')
const SDK_SPECIFIER = '@automatos/theme-sdk'
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

function builtNames(): string[] {
  return readdirSync(SURFACE_DIR)
    .filter((file) => file.endsWith('.json'))
    .map((file) => JSON.parse(readFileSync(path.join(SURFACE_DIR, file), 'utf8')) as SurfaceFile)
    .filter((surface) => surface.status === 'built')
    .flatMap((surface) => surface.names)
}

describe('PRD-261 A-10a — the SDK-context hello theme', () => {
  it('has its entry, manifest, stylesheet and preset', () => {
    expect(readdirSync(FIXTURE).sort()).toEqual(['entry.js', 'presets', 'theme.css', 'theme.json'])
    const manifest = JSON.parse(readFileSync(path.join(FIXTURE, 'theme.json'), 'utf8')) as Record<string, unknown>
    expect(manifest).toMatchObject({ entry: 'entry.js', styleSheet: 'theme.css', preset: 'presets/default.look.json', htmlClass: 'hello-a10a' })
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

  it('takes from the SDK only names a built area exports, useWorkspace among them', () => {
    const fromSdk = importsOf(ENTRY).filter((entry) => entry.specifier === SDK_SPECIFIER).flatMap((entry) => entry.names)
    const built = builtNames()
    expect(fromSdk).toContain('useWorkspace')
    expect(fromSdk.filter((name) => !built.includes(name))).toEqual([])
    expect(ENTRY).toMatch(/\buseWorkspace\(\)/)
  })
})
