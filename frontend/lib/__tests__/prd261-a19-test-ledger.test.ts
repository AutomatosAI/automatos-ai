// @vitest-environment node
/**
 * PRD-261 A-19f2 (plan §2.2.8) — the core test ledger: every test W-A8 took out of core says where
 * its assertions are now, so the move provably lost nothing.
 *
 * - fixtures/a19-test-paths-bcc3f79e5.json lists the test files, helpers and fixtures under the A-19
 *   paths at core main bcc3f79e5, and the ones added after it that W-A8 deletes too (`addedSince`),
 *   generated with python3 from `git ls-tree` (the command is in its header). CI checkouts are
 *   shallow, so this test reads that committed list and needs no git history.
 * - fixtures/a19-test-ledger.json has a row `{corePath, kitPath, coreReplacement}` for every listed
 *   path that is gone from the tree: `kitPath`, the test's home in the theme kit (a path in the kit
 *   repository; browser-e2e's package job checks it exists at KIT_REF, A-19f4), and/or
 *   `coreReplacement`, the core test that holds the assertion now. Paths are from the repository root.
 * - A row is for a gone path the list names, once; a named core replacement exists and is not gone
 *   itself; a kit path is a test file under the kit's themes/play/test, packages/<name>/test or tests folders
 *   (Play 1.1.1 moved its look and mascot tests into packages/look-kit and packages/mascot).
 */
import { existsSync, readFileSync } from 'fs'
import path from 'path'
import { describe, expect, it } from 'vitest'

interface PathsFile {
  readonly sha: string
  readonly command: string
  readonly dirs: readonly string[]
  readonly files: readonly string[]
  readonly paths: readonly string[]
  readonly addedSince: readonly string[]
}

interface LedgerRow {
  readonly corePath: string
  readonly kitPath: string | null
  readonly coreReplacement: string | null
}

const FIXTURES = path.join(__dirname, 'fixtures')
const REPO = path.resolve(__dirname, '../../..')
const PATHS_SHA = 'bcc3f79e525aeb3bf392dc3d65c608407fe80188'
const ROW_KEYS: readonly string[] = ['corePath', 'kitPath', 'coreReplacement']
/** Where the theme kit keeps tests: Play's own, and the kit's repository-wide ones. */
const KIT_TEST_PATH = /^(themes\/play\/test|packages\/[\w-]+\/test|tests)\/[\w./-]+\.(test\.tsx?|tsx?|json|css)$/
/** The paths list must hold at least this many (60 under the deleted folders at bcc3f79e5, and the rest). */
const PATHS_AT_LEAST = 80

function readJson<T>(name: string): T {
  return JSON.parse(readFileSync(path.join(FIXTURES, name), 'utf8')) as T
}

const PATHS = readJson<PathsFile>('a19-test-paths-bcc3f79e5.json')
const LEDGER = readJson<LedgerRow[]>('a19-test-ledger.json')
const LISTED: readonly string[] = [...PATHS.paths, ...PATHS.addedSince]
const gone = (rel: string): boolean => !existsSync(path.join(REPO, rel))

describe('PRD-261 A-19f2 — the committed paths list', () => {
  it('is the list at bcc3f79e5, made by the command in its header, over the A-19 paths it names', () => {
    expect(PATHS.sha).toBe(PATHS_SHA)
    expect(PATHS.command).toContain('ls-tree')
    expect(PATHS.paths.length).toBeGreaterThanOrEqual(PATHS_AT_LEAST)
    const underNamed = (rel: string) => PATHS.files.includes(rel) || PATHS.dirs.some((dir) => rel.startsWith(`${dir}/`))
    expect(LISTED.filter((rel) => !underNamed(rel))).toEqual([])
  })

  it('names each path once, and an added test only when bcc3f79e5 did not have it', () => {
    expect(LISTED.filter((rel, index) => LISTED.indexOf(rel) !== index)).toEqual([])
    expect(PATHS.addedSince.filter((rel) => PATHS.paths.includes(rel))).toEqual([])
  })
})

describe('PRD-261 A-19f2 — the ledger', () => {
  it('has a row, of the ledger format, for every listed path that is gone from the tree', () => {
    expect(LEDGER.filter((row) => Object.keys(row).sort().join() !== [...ROW_KEYS].sort().join())).toEqual([])
    const rowed = new Set(LEDGER.map((row) => row.corePath))
    expect(LISTED.filter((rel) => gone(rel) && !rowed.has(rel))).toEqual([])
  })

  it('rows only gone paths the list names, each once', () => {
    const paths = LEDGER.map((row) => row.corePath)
    expect(paths.filter((rel, index) => paths.indexOf(rel) !== index)).toEqual([])
    expect(paths.filter((rel) => !LISTED.includes(rel))).toEqual([])
    expect(paths.filter((rel) => !gone(rel))).toEqual([])
  })

  it('says where each one went: the kit, a core test, or both', () => {
    expect(LEDGER.filter((row) => row.kitPath === null && row.coreReplacement === null).map((row) => row.corePath)).toEqual([])
  })

  it('names core replacements that exist and are not themselves gone', () => {
    const replacements = LEDGER.flatMap((row) => (row.coreReplacement === null ? [] : [row.coreReplacement]))
    expect(replacements.filter((rel) => gone(rel) || LISTED.includes(rel))).toEqual([])
  })

  it('names kit paths that are test files in the kit, never a core path', () => {
    const kitPaths = LEDGER.flatMap((row) => (row.kitPath === null ? [] : [row.kitPath]))
    expect(kitPaths.filter((rel) => !KIT_TEST_PATH.test(rel) || rel.includes('..'))).toEqual([])
  })
})
