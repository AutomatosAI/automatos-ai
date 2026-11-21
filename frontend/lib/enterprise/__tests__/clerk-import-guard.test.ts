// @vitest-environment node
/**
 * PRD-277 US-501 — Clerk stays in the files that already held it.
 *
 * The enterprise edition mounts no ClerkProvider, so a component that imported `@clerk/*` itself
 * would throw there (or bundle Clerk into a page that never uses it). Components read identity
 * from lib/auth-hooks.ts, which hands back Clerk's hooks in saas and the API's who-am-I in
 * enterprise. This test walks the frontend's source and pins the files that import `@clerk/*` to
 * the ones that did before Wave 5; a new one is a decision, never a drift. No file this edition
 * adds may reach it at all, directly or through a shared file it imports (a side door).
 */
import { readdirSync, readFileSync, statSync } from 'fs'
import path from 'path'
import { describe, expect, it } from 'vitest'

const FRONTEND = path.resolve(__dirname, '..', '..', '..')
const SKIPPED_DIRS = new Set(['node_modules', '.next', '__tests__', 'e2e', 'public', 'coverage'])
const SOURCE = /\.(ts|tsx)$/
const TEST_FILE = /\.(test|spec)\.tsx?$/
const CLERK_IMPORT = /(?:from\s+|import\s*\(\s*|import\s+)['"]@clerk\//

/** The files that imported `@clerk/*` before PRD-277 Wave 5 (each saas-only or edition-gated). */
const CLERK_FILES = [
  'app/accept-invitation/page.tsx',
  'app/api/themes/files/[...path]/route.ts',
  'app/dev/reset-onboarding/page.tsx',
  'app/reset-password/page.tsx',
  'app/settings/profile/page.tsx',
  'app/sign-in/[[...rest]]/page.tsx',
  'app/sso-callback/page.tsx',
  'components/auth/sign-in-form.tsx',
  'components/auth/sign-up-form.tsx',
  'components/auth/user-profile-button.tsx',
  'components/clerk-api-client-provider.tsx',
  'components/providers.tsx',
  'lib/auth-hooks.ts',
  'lib/theme-host/api.ts',
  'lib/theme-host/boot-server.ts',
  'middleware.ts',
]

function sourceFiles(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const full = path.join(dir, name)
    if (statSync(full).isDirectory()) return SKIPPED_DIRS.has(name) || name.startsWith('.') ? [] : sourceFiles(full)
    return SOURCE.test(name) && !TEST_FILE.test(name) && !name.endsWith('.d.ts') ? [full] : []
  })
}

function clerkImporters(): string[] {
  return sourceFiles(FRONTEND)
    .filter((file) => CLERK_IMPORT.test(readFileSync(file, 'utf8')))
    .map((file) => path.relative(FRONTEND, file).split(path.sep).join('/'))
    .sort()
}

/**
 * The files this edition adds: every source file whose path names `enterprise` (a folder or the
 * file's own name), and these, which do not.
 */
const EDITION_FILES_NOT_NAMED = ['lib/edition-name.ts', 'lib/initials.ts', 'lib/public-routes.ts']
const NAMED_ENTERPRISE = /enterprise/i

// A value import or re-export (`import type` and `export type` bundle nothing), or a dynamic import().
const STATIC_IMPORT = /^\s*(?:import|export)\s+(?!type\s)(?:[^'"]*?\sfrom\s+)?['"]([^'"]+)['"]/gm
const DYNAMIC_IMPORT = /import\(\s*['"]([^'"]+)['"]\s*\)/g
const RESOLVED_SUFFIXES = ['', '.ts', '.tsx', '.js', '/index.ts', '/index.tsx']
const SOURCE_OR_SCRIPT = /\.(ts|tsx|js)$/

function relativeName(file: string): string {
  return path.relative(FRONTEND, file).split(path.sep).join('/')
}

function editionFiles(): string[] {
  const named = sourceFiles(FRONTEND).map(relativeName).filter((file) => NAMED_ENTERPRISE.test(file))
  return [...named, ...EDITION_FILES_NOT_NAMED].sort()
}

function specifiersOf(source: string): string[] {
  return [...source.matchAll(STATIC_IMPORT), ...source.matchAll(DYNAMIC_IMPORT)].map((match) => match[1])
}

/** The frontend file a local specifier names (`@/…` or relative), or null for a package. */
function resolveLocal(specifier: string, from: string): string | null {
  const base = specifier.startsWith('@/')
    ? path.join(FRONTEND, specifier.slice(2))
    : specifier.startsWith('.') ? path.resolve(path.dirname(from), specifier) : null
  if (base === null) return null
  const found = RESOLVED_SUFFIXES.map((suffix) => base + suffix).find((candidate) => {
    try {
      return statSync(candidate).isFile()
    } catch {
      return false
    }
  })
  return found && SOURCE_OR_SCRIPT.test(found) ? found : null
}

/** Each file `start` reaches through its value imports that imports `@clerk/*`, with the path to it. */
function clerkReachedFrom(start: string): string[] {
  const seen = new Set<string>()
  const queue: Array<{ file: string; trail: string[] }> = [{ file: path.join(FRONTEND, start), trail: [start] }]
  const reached: string[] = []
  while (queue.length > 0) {
    const { file, trail } = queue.shift() as { file: string; trail: string[] }
    if (seen.has(file)) continue
    seen.add(file)
    const specifiers = specifiersOf(readFileSync(file, 'utf8'))
    if (specifiers.some((specifier) => specifier.startsWith('@clerk/'))) reached.push(trail.join(' → '))
    specifiers
      .map((specifier) => resolveLocal(specifier, file))
      .filter((next): next is string => next !== null && !seen.has(next))
      .forEach((next) => queue.push({ file: next, trail: [...trail, relativeName(next)] }))
  }
  return reached
}

describe('PRD-277 US-501 — no component outside the Clerk files imports @clerk/*', () => {
  const importers = clerkImporters()

  it('finds Clerk only in the files that held it before', () => {
    expect(importers).toEqual([...CLERK_FILES].sort())
  })

  it('never imports Clerk from a file this edition adds, directly or through any file it imports', () => {
    const files = editionFiles()
    expect(files).toEqual(expect.arrayContaining([
      'lib/enterprise/session-fetch.ts', 'components/team/enterprise-member-actions.tsx', 'app/settings/profile/enterprise-profile.tsx',
    ]))
    expect(files.flatMap(clerkReachedFrom)).toEqual([])
  })

  it('follows imports the way the bundler does: a side door through a shared file is found', () => {
    // use-team-members.ts imports no Clerk itself; it reaches it through lib/theme-host/api.ts.
    expect(clerkReachedFrom('components/team/use-team-members.ts')).toContain(
      'components/team/use-team-members.ts → lib/theme-host/api.ts',
    )
    expect(specifiersOf("import type { X } from '@clerk/nextjs'\nexport type { Y } from '@clerk/shared'")).toEqual([])
    expect(specifiersOf("import {\n  a,\n} from '@/lib/x'\nexport { b } from './y'\nconst m = import('./z')")).toEqual([
      '@/lib/x', './y', './z',
    ])
  })

  it('reads the enterprise branch of every shared Clerk file from the edition, not from Clerk', () => {
    for (const file of ['components/providers.tsx', 'lib/auth-hooks.ts', 'middleware.ts', 'lib/theme-host/api.ts']) {
      expect(readFileSync(path.join(FRONTEND, file), 'utf8'), file).toMatch(/isEnterprise/)
    }
  })
})
