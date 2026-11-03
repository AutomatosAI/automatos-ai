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
 *
 * PRD-261 M-01 (host 1.1.0): the manifests section carries theme.json's mobile block, and the
 * block's lists (the token list, the font roles, the mascot modes, the contrast pairs and their AA
 * minimums, the lowest host) equal contract.py's MOBILE_* names.
 *
 * PRD-261 A-20 (QB-11): htmlClass 'play' is no longer reserved. The corpus gives it to any theme id
 * and still refuses every class the host keeps, and neither contract gives a class to an id.
 *
 * The browser's first-paint gate (boot-sanitize.ts, through css-value.ts isBootValue): the prepaintDrops
 * section's cases give the server reader's verdicts and sentences here too, and the gate's functions and
 * the longest value equal css_value.py's and contract.py's. The firstPaint section's cases give
 * first_paint.py's verdicts and sentences through first-paint.ts, which asks the browser's own
 * sanitizePrepaint and lookSpecOf what a theme's table and looks keep.
 */
import { readFileSync } from 'fs'
import path from 'path'
import { describe, expect, it } from 'vitest'

import { BOOT_PRESET_MAX_BYTES, BOOT_PRESETS_MAX } from '../boot-sanitize'
import { HTML_CLASS_PATTERN } from '../constants'
import {
  BOOT_VALUE_FUNCTIONS, bootValueProblem, cssValueProblem, PREPAINT_ERROR_CODES, PREPAINT_VALUE_MAX, prepaintDropErrors,
  prepaintErrors, TONE_RULE_FIELDS, TONE_RULE_KEY, TONE_RULE_TONES, type ThemeError,
} from '../css-value'
import { FIRST_PAINT_ERROR_CODES, firstPaintErrors } from '../first-paint'
import { LOOK_ERROR_CODES, validateLook, validateLookSchema, type LookSchema } from '../look-schema'
import {
  MOBILE_AA_TEXT, MOBILE_AA_UI, MOBILE_CONTRAST_PAIRS, MOBILE_FONT_ROLES, MOBILE_MASCOT_MODES, MOBILE_MIN_HOST,
  MOBILE_REQUIRED_TOKENS, MOBILE_TOKEN_NAMES,
} from '../mobile-manifest'
import {
  HOST_OWNED_ROUTES, KNOWN_SLOTS, MANIFEST_ERROR_CODES, PAGE_SCOPE, PAGE_SOURCES, REQUIRED_SLOTS,
  RESERVED_HTML_CLASSES, THEME_BASES, THEME_HOST_VERSION, THEME_TONES, validateManifest,
} from '../package-manifest'
import { SDK_VERSION } from '../sdk'
import sharedModules from '../shared-modules.json'

const REPO_ROOT = path.resolve(__dirname, '../../../..')
const THEMES_DIR = path.join(REPO_ROOT, 'orchestrator/services/themes')
const CORPUS_PATH = path.join(REPO_ROOT, 'orchestrator/tests/fixtures/themes/cases.json')
// PB-3: the host contract's version W-E3 ships; a later bump edits this with contract.py and shared-modules.json.
const CONTRACT_VERSION = '1.1.0'
// Theme voice (C5): the SDK moved to 1.2.0 for its voice area; PRD-268 (K12) to 1.3.0 for its proactive area; PRD-268
// Wave K (K17) to 1.4.0 for a message's parts, the brief's speaking row and its thumbs. The host contract stayed at
// CONTRACT_VERSION.
const SDK_CONTRACT_VERSION = '1.4.0'
// PB-32: the mobile block joined host 1.1.0, so a theme with one admits no older host (a later bump keeps this).
const MOBILE_BLOCK_HOST = '1.1.0'
// A-20 (QB-11): the <html> classes the host keeps, and the one it no longer reserves for any id.
const HOST_HTML_CLASSES: readonly string[] = ['studio', 'classic', 'dark', 'light', 'system']
const FREED_HTML_CLASS = 'play'
const RESERVED_CODE = 'RESERVED_HTML_CLASS'
const FREED_CLASS_IDS_MIN = 3

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
  readonly prepaintDrops: CorpusSection
  readonly firstPaint: CorpusSection
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

/** The manifest cases that set `htmlClass` to `htmlClass` (A-20's reserved-class cases). */
function htmlClassCases(htmlClass: string): CorpusCase[] {
  return CORPUS.manifests.cases.filter((testCase) => testCase.set?.htmlClass === htmlClass)
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

  it.each([...CORPUS.prepaintDrops.cases])('prepaint drops: $name', (testCase) => {
    const errors = prepaintDropErrors(testCase.document)
    expectVerdict(errors, testCase.code, PREPAINT_ERROR_CODES)
    if (testCase.messages !== undefined) expect(errors.map((error) => error.message)).toEqual(testCase.messages)
  })

  it.each([...CORPUS.firstPaint.cases])('first paint: $name', (testCase) => {
    const errors = firstPaintErrors(helloLookSchema(), testCase.document)
    expectVerdict(errors, testCase.code, FIRST_PAINT_ERROR_CODES)
    if (testCase.messages !== undefined) expect(errors.map((error) => error.message)).toEqual(testCase.messages)
  })

  it('pins the sentence of every refused drop case, and has a case the first paint keeps', () => {
    const refused = [...CORPUS.prepaintDrops.cases, ...CORPUS.firstPaint.cases].filter((testCase) => testCase.code !== null)
    expect(refused.length).toBeGreaterThan(0)
    refused.forEach((testCase) => expect(testCase.messages?.length, testCase.name).toBeGreaterThan(0))
    expect(CORPUS.prepaintDrops.cases.some((testCase) => testCase.code === null && 'document' in testCase)).toBe(true)
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
      expect(() => prepaintDropErrors(value)).not.toThrow()
      expect(() => firstPaintErrors(helloLookSchema(), value)).not.toThrow()
      expect(() => cssValueProblem(value)).not.toThrow()
      expect(() => bootValueProblem(value)).not.toThrow()
    })
  })

  it('reads a field only when the document has it, never one it inherits', () => {
    const base = CORPUS.manifests.base ?? {}
    const withoutSlots = Object.fromEntries(Object.entries(base).filter(([key]) => key !== 'slots'))
    const inherited: unknown = Object.assign(Object.create({ slots: base.slots }), withoutSlots)
    expect(validateManifest(inherited).errors.map((error) => error.code)).toEqual(['MISSING_FIELD'])
  })

  it("gives htmlClass 'play' to any theme id and refuses each class the host keeps (A-20)", () => {
    const freed = htmlClassCases(FREED_HTML_CLASS)
    const ids = new Set(freed.map((testCase) => testCase.set?.id ?? CORPUS.manifests.base?.id))
    expect(ids.size).toBeGreaterThanOrEqual(FREED_CLASS_IDS_MIN)
    expect(freed.map((testCase) => testCase.code)).toEqual(freed.map(() => null))
    HOST_HTML_CLASSES.forEach((htmlClass) => {
      const codes = htmlClassCases(htmlClass).map((testCase) => testCase.code)
      expect(codes.length, htmlClass).toBeGreaterThan(0)
      expect(Array.from(new Set(codes)), htmlClass).toEqual([RESERVED_CODE])
    })
  })

  it('runs every section the browser shares with the server', () => {
    const sections = [CORPUS.manifests, CORPUS.lookSchemas, CORPUS.looks, CORPUS.prepaint, CORPUS.prepaintDrops, CORPUS.firstPaint]
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

  it('is at host 1.1.0 (the boot tone rule, PB-3) and SDK 1.4.0 (message parts, the speaking row, the thumbs: PRD-268 K17)', () => {
    expect(THEME_HOST_VERSION).toBe(CONTRACT_VERSION)
    expect(SDK_VERSION).toBe(SDK_CONTRACT_VERSION)
  })

  it('reserves the same <html> classes, and neither side gives one to a theme id (A-20)', () => {
    expect(strings('RESERVED_HTML_CLASSES')).toEqual([...RESERVED_HTML_CLASSES])
    expect(RESERVED_HTML_CLASSES).toEqual(HOST_HTML_CLASSES)
    expect(PYTHON.contract).not.toMatch(/\bHTML_CLASS_OWNERS\b/)
    expect(PYTHON.manifest).not.toMatch(/\bHTML_CLASS_OWNERS\b/)
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

  it("has the same first-paint gate: its functions and the longest prepaint value (boot-sanitize.ts's)", () => {
    expect(strings('BOOT_VALUE_FUNCTIONS', PYTHON.cssValue)).toEqual([...BOOT_VALUE_FUNCTIONS])
    expect(Number(pythonValue(PYTHON.contract, 'PREPAINT_VALUE_MAX'))).toBe(PREPAINT_VALUE_MAX)
    expect(PREPAINT_VALUE_MAX).toBe(200)
    expect(BOOT_VALUE_FUNCTIONS.has('linear-gradient')).toBe(false)
  })

  it('has the same mobile block lists, AA minimums and lowest host (M-01, plan §2.2.11)', () => {
    expect(strings('MOBILE_TOKEN_NAMES')).toEqual([...MOBILE_TOKEN_NAMES])
    expect(strings('MOBILE_REQUIRED_TOKENS')).toEqual([...MOBILE_REQUIRED_TOKENS])
    expect(strings('MOBILE_FONT_ROLES')).toEqual([...MOBILE_FONT_ROLES])
    expect(strings('MOBILE_MASCOT_MODES')).toEqual([...MOBILE_MASCOT_MODES])
    expect(strings('MOBILE_CONTRAST_PAIRS')).toEqual(MOBILE_CONTRAST_PAIRS.flat())
    expect(Number(pythonValue(PYTHON.contract, 'MOBILE_AA_TEXT'))).toBe(MOBILE_AA_TEXT)
    expect(Number(pythonValue(PYTHON.contract, 'MOBILE_AA_UI'))).toBe(MOBILE_AA_UI)
    expect([MOBILE_AA_TEXT, MOBILE_AA_UI]).toEqual([4.5, 3])
    expect(strings('MOBILE_MIN_HOST')).toEqual([MOBILE_MIN_HOST])
    expect(MOBILE_MIN_HOST).toBe(MOBILE_BLOCK_HOST)
  })

  it('runs a mobile case of every verdict the block has (M-01)', () => {
    const mobileCases = CORPUS.manifests.cases.filter((testCase) => testCase.name.startsWith('mobile'))
    const codes = new Set(mobileCases.map((testCase) => testCase.code))
    expect(codes).toEqual(new Set([null, 'BAD_MOBILE', 'UNKNOWN_FIELD']))
  })

  it('knows the same error codes as each Python validator', () => {
    expect(strings('MANIFEST_ERROR_CODES', PYTHON.manifest)).toEqual([...MANIFEST_ERROR_CODES])
    expect(strings('LOOK_ERROR_CODES', PYTHON.lookSchema)).toEqual([...LOOK_ERROR_CODES])
    expect(strings('PREPAINT_ERROR_CODES', PYTHON.cssValue)).toEqual([...PREPAINT_ERROR_CODES])
  })
})
