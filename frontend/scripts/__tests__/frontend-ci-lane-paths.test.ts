// @vitest-environment node
/**
 * PRD-261: test.yml starts frontend-ci for a pull request only when one of its files matches
 * FRONTEND_PATHS. Several frontend tests read server files as text so that drift fails them:
 * scan-report.test.ts reads review_stage.py and scan.py, appearance-tab.test.tsx reads optin.py,
 * validator-parity.test.ts the validators, the api-client theme tests the dicts their types
 * mirror. A pull request that changes only such a file has to start frontend-ci too. Otherwise
 * the drift turns up later, as an unrelated red on the next pull request that touches frontend/.
 *
 * - Every orchestrator/services/themes module a frontend test names in a quoted string, as
 *   scan-report.test.ts names review_stage.py, matches FRONTEND_PATHS, so a test that starts
 *   reading another module fails here until the lane lists it. This file names none that way, so
 *   the scan cannot find a module here alone.
 * - The other server files the frontend tests read, by path segments that no literal scan sees,
 *   match it too.
 */
import { existsSync, readdirSync, readFileSync } from 'fs'
import path from 'path'
import { describe, expect, it } from 'vitest'

const FRONTEND = path.resolve(__dirname, '..', '..')
const REPO_ROOT = path.resolve(FRONTEND, '..')
const TEST_YML = path.join(REPO_ROOT, '.github', 'workflows', 'test.yml')
const THEMES_PY = 'orchestrator/services/themes'
const LANE_LINE = /^ +FRONTEND_PATHS: (.+)$/m
const PY_LITERAL = /['"](\w+\.py)['"]/g
const SOURCE_FILE = /\.(ts|tsx|js|mjs)$/
const TEST_FILE = /\.test\.[jt]sx?$/
const TESTS_DIR = '__tests__'
/** Modules scan-report.test.ts and appearance-tab.test.tsx read: the scan must keep finding them. */
const KNOWN_MODULE_STEMS = ['review_stage', 'scan', 'optin']
/** A theme module no frontend test reads, which the lane must not match. */
const UNREAD_MODULE = `${THEMES_PY}/not_read_by_the_frontend.py`
/** Read by path segments; the readers are named beside each. */
const OTHER_SERVER_FILES = [
  'orchestrator/reports/route-manifest.json', // scripts/check-route-contract.js, the api-client route tests
  'orchestrator/contracts/page-manifest.json', // lib/__tests__/page-manifest-contract.test.ts
  'orchestrator/tests/fixtures/themes/cases.json', // validator-parity.test.ts, contrast-parity.test.ts
  'orchestrator/services/themes/data/phone-layouts.json', // validator-parity.test.ts (PRD-271: the browser's copy)
  'orchestrator/tests/fixtures/themes/look-storage.json', // lib/theme-host/looks/__tests__/hello-looks.ts
  'orchestrator/core/auth/step_up.py', // lib/theme-host/__tests__/api-hint.test.ts
  'orchestrator/core/cli_presets.py', // lib/__tests__/session-clis.test.ts
  'e2e/browser/fixtures/hello-a09b/boot.json', // boot-server.test.ts, hello-looks.ts
  'e2e/browser/fixtures/hello-a04d/entry.js', // app/__tests__/theme-page-layouts.test.tsx
  'e2e/browser/fixtures/hello-a04d/entry.frame-only.js', // app/__tests__/theme-page-layouts.test.tsx
]

/** FRONTEND_PATHS as test.yml's changes job greps it (an ERE that JavaScript reads the same way). */
function frontendLane(): RegExp {
  const source = LANE_LINE.exec(readFileSync(TEST_YML, 'utf8'))?.[1]
  if (source === undefined || !source.startsWith('^')) {
    throw new Error('test.yml must set FRONTEND_PATHS on one plain line')
  }
  return new RegExp(source)
}

/** Every test source under dir: *.test.* files and anything in a __tests__ folder. */
function testSources(dir: string): string[] {
  return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const full = path.join(dir, entry.name)
    if (entry.isDirectory()) {
      return entry.name === 'node_modules' || entry.name.startsWith('.') ? [] : testSources(full)
    }
    const isTest = TEST_FILE.test(entry.name) || dir.split(path.sep).includes(TESTS_DIR)
    return SOURCE_FILE.test(entry.name) && isTest ? [full] : []
  })
}

/** The theme modules the frontend tests name, as repository paths. */
function themeModulesNamed(): string[] {
  const named = testSources(FRONTEND).flatMap((file) =>
    [...readFileSync(file, 'utf8').matchAll(PY_LITERAL)].map((match) => `${THEMES_PY}/${match[1]}`),
  )
  return [...new Set(named)].filter((module) => existsSync(path.join(REPO_ROOT, module))).sort()
}

describe('PRD-261: a server file a frontend test reads starts frontend-ci', () => {
  it('finds the theme modules the drift tests read', () => {
    const known = KNOWN_MODULE_STEMS.map((stem) => `${THEMES_PY}/${stem}.py`)
    expect(themeModulesNamed()).toEqual(expect.arrayContaining(known))
  })

  it('FRONTEND_PATHS lists every theme module a frontend test names', () => {
    const lane = frontendLane()
    expect(themeModulesNamed().filter((module) => !lane.test(module))).toEqual([])
    expect(lane.test(UNREAD_MODULE)).toBe(false)
  })

  it('FRONTEND_PATHS lists the other server files the frontend tests read', () => {
    const lane = frontendLane()
    expect(OTHER_SERVER_FILES.filter((file) => !existsSync(path.join(REPO_ROOT, file)))).toEqual([])
    expect(OTHER_SERVER_FILES.filter((file) => !lane.test(file))).toEqual([])
  })
})
