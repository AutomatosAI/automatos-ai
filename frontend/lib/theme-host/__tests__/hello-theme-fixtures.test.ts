/**
 * PRD-261 A-01 — the hello theme the browser lane serves (e2e/browser/fixtures/hello-theme) is
 * valid as files: every .js parses as an ES module, and every .json parses. The lane type-checks
 * the theme against @types/react 19 (npx tsc -p fixtures/hello-theme/tsconfig.json).
 */
import { readdirSync, readFileSync } from 'fs'
import path from 'path'
import ts from 'typescript'
import { describe, expect, it } from 'vitest'

const FIXTURES = path.resolve(__dirname, '../../../../e2e/browser/fixtures/hello-theme')

function filesUnder(dir: string): string[] {
  return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const full = path.join(dir, entry.name)
    return entry.isDirectory() ? filesUnder(full) : [full]
  })
}

function syntaxErrors(source: string, fileName: string): string[] {
  const output = ts.transpileModule(source, {
    reportDiagnostics: true,
    fileName,
    compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022, allowJs: true },
  })
  return (output.diagnostics ?? []).map((diagnostic) => ts.flattenDiagnosticMessageText(diagnostic.messageText, '\n'))
}

const FILES = filesUnder(FIXTURES).map((file) => path.relative(FIXTURES, file))
const SCRIPTS = FILES.filter((file) => file.endsWith('.js'))
const JSON_FILES = FILES.filter((file) => file.endsWith('.json'))

describe('PRD-261 A-01 — the hello theme fixtures', () => {
  it('has the entries, manifests and preset the lane serves', () => {
    expect([...SCRIPTS].sort()).toEqual(['entry.js', 'entry.render-throws.js', 'entry.throws.js'])
    expect(JSON_FILES).toEqual(expect.arrayContaining(['theme.json', 'theme.engines-mismatch.json', path.join('presets', 'default.look.json')]))
  })

  it.each(SCRIPTS)('%s parses as an ES module', (file) => {
    expect(syntaxErrors(readFileSync(path.join(FIXTURES, file), 'utf8'), file)).toEqual([])
  })

  it.each(JSON_FILES)('%s parses as JSON', (file) => {
    expect(() => JSON.parse(readFileSync(path.join(FIXTURES, file), 'utf8'))).not.toThrow()
  })
})
