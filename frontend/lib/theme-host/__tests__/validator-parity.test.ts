/**
 * PRD-261 A-06b — the browser's theme validators give the server's verdicts.
 *
 * Every case of the shared corpus (orchestrator/tests/fixtures/themes/cases.json, which
 * orchestrator/tests/test_prd261_theme_package.py runs against the Python validators) gives the
 * same verdict and code here: a case's code is null for a valid input, otherwise every error
 * carries that code. cssScope and packages are the server's alone (the zip reader and the CSS
 * scope check). The host's promises (the reserved classes, the page scope and sources, the slots,
 * the host and SDK versions, the error codes) are read from the Python sources as text and must
 * equal the TypeScript ones, so neither side can change one alone.
 *
 * PRD-261 A-19c: the prepaint section carries the tone rule's cases, and the ones with `messages`
 * pin the exact sentences, which the server gives too; the tone rule's names and the boot's
 * preset caps equal contract.py's.
 */
import { readFileSync } from 'fs'
import path from 'path'
import { describe, expect, it } from 'vitest'

import { BOOT_PRESET_MAX_BYTES, BOOT_PRESETS_MAX } from '../boot-sanitize'
import { HTML_CLASS_PATTERN } from '../constants'
import {
  cssValueProblem, PREPAINT_ERROR_CODES, prepaintErrors, TONE_RULE_FIELDS, TONE_RULE_KEY, TONE_RULE_TONES, type ThemeError,
} from '../css-value'
import { LOOK_ERROR_CODES, validateLook, validateLookSchema, type LookSchema } from '../look-schema'
import {
  HOST_OWNED_ROUTES, HTML_CLASS_OWNERS, KNOWN_SLOTS, MANIFEST_ERROR_CODES, PAGE_SCOPE, PAGE_SOURCES, REQUIRED_SLOTS,
  RESERVED_HTML_CLASSES, THEME_BASES, THEME_HOST_VERSION, THEME_TONES, validateManifest,
} from '../package-manifest'
import { SDK_VERSION } from '../sdk'
import sharedModules from '../shared-modules.json'

const REPO_ROOT = path.resolve(__dirname, '../../../..')
const THEMES_DIR = path.join(REPO_ROOT, 'orchestrator/services/themes')
const CORPUS_PATH = path.join(REPO_ROOT, 'orchestrator/tests/fixtures/themes/cases.json')
// PB-3: the host and SDK version W-E3 ships; a later bump edits this with contract.py and shared-modules.json.
const CONTRACT_VERSION = '1.1.0'

interface CorpusCase {
  readonly name: string
  readonly code: string | null
  readonly set?: Readonly<Record<string, unknown>>
  readonly unset?: readonly string[]
  readonly document?: unknown
  /** A-19c: the exact sentences both languages give, in order. */
  readonly messages?: readonly string[]
}

interface CorpusSection {
  readonly base?: Readonly<Record<string, unknown>>
  readonly schema?: unknown
  readonly cases: readonly CorpusCase[]
}

interface Corpus {
  readonly manifests: CorpusSection
  readonly lookSchemas: CorpusSection
  readonly looks: CorpusSection
  readonly prepaint: CorpusSection
  readonly cssValues: { readonly valid: readonly string[]; readonly invalid: readonly string[] }
}

const CORPUS = JSON.parse(readFileSync(CORPUS_PATH, 'utf8')) as Corpus
const PYTHON = {
  contract: readFileSync(path.join(THEMES_DIR, 'contract.py'), 'utf8'),
  manifest: readFileSync(path.join(THEMES_DIR, 'manifest.py'), 'utf8'),
  lookSchema: readFileSync(path.join(THEMES_DIR, 'look_schema.py'), 'utf8'),
  cssValue: readFileSync(path.join(THEMES_DIR, 'css_value.py'), 'utf8'),
}
const OPENERS = '([{'
const CLOSERS = ')]}'

/** The right-hand side of the module-level assignment to `name`: a bracketed literal, or the rest of its line. */
function pythonValue(source: string, name: string): string {
  const start = new RegExp(`^${name}\\b[^=\\n]*=[ ]*`, 'm').exec(source)
  if (start === null) throw new Error(`${name} is not assigned at the top level of the Python source`)
  const from = start.index + start[0].length
  if (!OPENERS.includes(source[from])) return source.slice(from, source.indexOf('\n', from))
  let depth = 0
  for (let at = from; at < source.length; at += 1) {
    if (OPENERS.includes(source[at])) depth += 1
    if (CLOSERS.includes(source[at])) depth -= 1
    if (depth === 0) return source.slice(from, at + 1)
  }
  throw new Error(`${name}'s value never closes`)
}

/** Every double-quoted string in a Python literal, in order. */
function pythonStrings(literal: string): string[] {
  return Array.from(literal.matchAll(/"((?:[^"\\]|\\.)*)"/g), (match) => match[1])
}

/** PAGE_SOURCES-shaped dict literal: `{"route": ("file", ...), ...}`. */
function pythonTupleDict(literal: string): Record<string, string[]> {
  const entries = Array.from(literal.matchAll(/"([^"]+)":\s*\(([^)]*)\)/g), (match): [string, string[]] => [
    match[1], pythonStrings(match[2]),
  ])
  return Object.fromEntries(entries)
}

/** The case's input: `document` as given, or `base` with `set` applied and `unset` removed. */
function patched(base: Readonly<Record<string, unknown>> | undefined, testCase: CorpusCase): unknown {
  if ('document' in testCase) return testCase.document
  const merged: Record<string, unknown> = { ...base, ...testCase.set }
  const unset = testCase.unset ?? []
  return Object.fromEntries(Object.entries(merged).filter(([key]) => !unset.includes(key)))
}

function expectPlain(errors: readonly ThemeError[], known: readonly string[]): void {
  errors.forEach((error) => {
    expect(Object.keys(error).sort()).toEqual(['code', 'message', 'path'])
    expect(known).toContain(error.code)
    expect(error.message.trim()).not.toBe('')
    expect(error.message).not.toContain('\n')
  })
}

function expectVerdict(errors: readonly ThemeError[], code: string | null, known: readonly string[]): void {
  expectPlain(errors, known)
  const codes = errors.map((error) => error.code)
  if (code === null) {
    expect(errors).toEqual([])
  } else {
    expect(codes.length).toBeGreaterThan(0)
    expect(Array.from(new Set(codes))).toEqual([code])
  }
}

function isToneCase(document: unknown): boolean {
  return typeof document === 'object' && document !== null && !Array.isArray(document) && TONE_RULE_KEY in document
}

function helloLookSchema(): LookSchema {
  const { schema, errors } = validateLookSchema(CORPUS.looks.schema)
  expect(errors).toEqual([])
  if (schema === null) throw new Error('the corpus look schema must be valid')
  return schema
}

describe('PRD-261 A-06b — the TypeScript validators give the corpus verdicts', () => {
  it.each([...CORPUS.manifests.cases])('manifest: $name', (testCase) => {
    const result = validateManifest(patched(CORPUS.manifests.base, testCase))
    expectVerdict(result.errors, testCase.code, MANIFEST_ERROR_CODES)
    expect(result.manifest === null).toBe(testCase.code !== null)
  })

  it.each([...CORPUS.lookSchemas.cases])('look schema: $name', (testCase) => {
    const result = validateLookSchema(testCase.document)
    expectVerdict(result.errors, testCase.code, LOOK_ERROR_CODES)
    expect(result.schema === null).toBe(testCase.code !== null)
  })

  it.each([...CORPUS.looks.cases])('look: $name', (testCase) => {
    const errors = validateLook(helloLookSchema(), patched(CORPUS.looks.base, testCase))
    expectVerdict(errors, testCase.code, LOOK_ERROR_CODES)
  })

  it.each([...CORPUS.prepaint.cases])('prepaint: $name', (testCase) => {
    const errors = prepaintErrors(testCase.document)
    expectVerdict(errors, testCase.code, PREPAINT_ERROR_CODES)
    if (testCase.messages !== undefined) expect(errors.map((error) => error.message)).toEqual(testCase.messages)
  })

  it('has a valid tone rule and pins the sentence of each kind of refusal (A-19c)', () => {
    const toneCases = CORPUS.prepaint.cases.filter((testCase) => isToneCase(testCase.document))
    expect(toneCases.some((testCase) => testCase.code === null)).toBe(true)
    const pinned = toneCases.flatMap((testCase) => testCase.messages ?? [])
    expect(pinned.length).toBeGreaterThanOrEqual(8)
    pinned.forEach((message) => expect(message.startsWith('boot/prepaint.json#tone')).toBe(true))
  })

  it.each([...CORPUS.cssValues.valid])('CSS value %j passes', (value) => {
    expect(cssValueProblem(value)).toBeNull()
  })

  it.each([...CORPUS.cssValues.invalid])('CSS value %j is refused with a reason', (value) => {
    const problem = cssValueProblem(value)
    expect(typeof problem).toBe('string')
    expect(problem).not.toBe('')
    expect(problem).not.toContain('\n')
  })

  it('never throws on input of the wrong shape', () => {
    const odd: unknown[] = [undefined, null, 0, 'x', [], {}, Object.create(null), { constructor: 1, toString: 'x' }]
    odd.forEach((value) => {
      expect(() => validateManifest(value)).not.toThrow()
      expect(() => validateLookSchema(value)).not.toThrow()
      expect(() => validateLook(helloLookSchema(), value)).not.toThrow()
      expect(() => prepaintErrors(value)).not.toThrow()
      expect(() => cssValueProblem(value)).not.toThrow()
    })
  })

  it('reads a field only when the document has it, never one it inherits', () => {
    const base = CORPUS.manifests.base ?? {}
    const withoutSlots = Object.fromEntries(Object.entries(base).filter(([key]) => key !== 'slots'))
    const inherited: unknown = Object.assign(Object.create({ slots: base.slots }), withoutSlots)
    expect(validateManifest(inherited).errors.map((error) => error.code)).toEqual(['MISSING_FIELD'])
  })

  it('runs every section the browser shares with the server', () => {
    const sections = [CORPUS.manifests, CORPUS.lookSchemas, CORPUS.looks, CORPUS.prepaint]
    sections.forEach((section) => expect(section.cases.length).toBeGreaterThan(0))
    expect(CORPUS.cssValues.valid.length).toBeGreaterThan(0)
    expect(CORPUS.cssValues.invalid.length).toBeGreaterThan(0)
  })
})

describe('PRD-261 A-06b — the host contract equals contract.py', () => {
  const strings = (name: string, source = PYTHON.contract): string[] => pythonStrings(pythonValue(source, name))

  it('has the same host and SDK versions, and shared-modules.json carries both (PA-2)', () => {
    expect(strings('THEME_HOST_VERSION')).toEqual([THEME_HOST_VERSION])
    expect(strings('SDK_VERSION')).toEqual([SDK_VERSION])
    expect(sharedModules.hostVersion).toBe(THEME_HOST_VERSION)
    expect(sharedModules.sdkVersion).toBe(SDK_VERSION)
  })

  it('is at 1.1.0: the storage area and the boot tone rule are new (PB-3, D7 QB-3)', () => {
    expect(THEME_HOST_VERSION).toBe(CONTRACT_VERSION)
    expect(SDK_VERSION).toBe(CONTRACT_VERSION)
  })

  it('reserves the same <html> classes, with Play keeping html.play (PA-8)', () => {
    expect(strings('RESERVED_HTML_CLASSES')).toEqual([...RESERVED_HTML_CLASSES])
    expect(strings('HTML_CLASS_OWNERS')).toEqual(Array.from(HTML_CLASS_OWNERS).flat())
    expect(strings('HTML_CLASS_PATTERN')).toEqual([HTML_CLASS_PATTERN.source.replace(/^\^|\$$/g, '')])
  })

  it('serves the same slots and requires the same ones (PA-14)', () => {
    expect(strings('KNOWN_SLOTS')).toEqual([...KNOWN_SLOTS])
    expect(strings('REQUIRED_SLOTS')).toEqual([...REQUIRED_SLOTS])
    expect(strings('THEME_BASES')).toEqual([...THEME_BASES])
    expect(strings('THEME_TONES')).toEqual([...THEME_TONES])
  })

  it('has the same page scope, page sources and host-owned routes (Q3)', () => {
    expect(strings('PAGE_SCOPE')).toEqual([...PAGE_SCOPE])
    expect(pythonTupleDict(pythonValue(PYTHON.contract, 'PAGE_SOURCES'))).toEqual(PAGE_SOURCES)
    expect(strings('HOST_OWNED_ROUTES')).toEqual([...HOST_OWNED_ROUTES])
  })

  it('has the same tone rule names and boot preset caps (A-19c, plan §2.2.3)', () => {
    expect(strings('TONE_RULE_KEY')).toEqual([TONE_RULE_KEY])
    expect(strings('TONE_RULE_FIELDS')).toEqual([...TONE_RULE_FIELDS])
    expect(strings('TONE_RULE_TONES')).toEqual([...TONE_RULE_TONES])
    expect(Number(pythonValue(PYTHON.contract, 'BOOT_PRESETS_MAX'))).toBe(BOOT_PRESETS_MAX)
    expect(Number(pythonValue(PYTHON.contract, 'BOOT_PRESET_MAX_BYTES'))).toBe(BOOT_PRESET_MAX_BYTES)
    expect([BOOT_PRESETS_MAX, BOOT_PRESET_MAX_BYTES]).toEqual([12, 4096])
  })

  it('knows the same error codes as each Python validator', () => {
    expect(strings('MANIFEST_ERROR_CODES', PYTHON.manifest)).toEqual([...MANIFEST_ERROR_CODES])
    expect(strings('LOOK_ERROR_CODES', PYTHON.lookSchema)).toEqual([...LOOK_ERROR_CODES])
    expect(strings('PREPAINT_ERROR_CODES', PYTHON.cssValue)).toEqual([...PREPAINT_ERROR_CODES])
  })
})
