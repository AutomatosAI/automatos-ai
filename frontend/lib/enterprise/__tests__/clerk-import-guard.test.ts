// @vitest-environment node
/**
 * PRD-277 US-501 — Clerk stays in the files that already held it.
 *
 * The enterprise edition mounts no ClerkProvider, so a component that imported `@clerk/*` itself
 * would throw there (or bundle Clerk into a page that never uses it). Components read identity
 * from lib/auth-hooks.ts, which hands back Clerk's hooks in saas and the API's who-am-I in
 * enterprise. This test walks the frontend's source and pins the files that import `@clerk/*` to
 * the ones that did before Wave 5; a new one is a decision, never a drift. Nothing under
 * lib/enterprise or components/enterprise may import it at all.
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

describe('PRD-277 US-501 — no component outside the Clerk files imports @clerk/*', () => {
  const importers = clerkImporters()

  it('finds Clerk only in the files that held it before', () => {
    expect(importers).toEqual([...CLERK_FILES].sort())
  })

  it('never imports Clerk from the enterprise modules', () => {
    expect(importers.filter((file) => /(^|\/)enterprise\//.test(file))).toEqual([])
  })

  it('reads the enterprise branch of every shared Clerk file from the edition, not from Clerk', () => {
    for (const file of ['components/providers.tsx', 'lib/auth-hooks.ts', 'middleware.ts', 'lib/theme-host/api.ts']) {
      expect(readFileSync(path.join(FRONTEND, file), 'utf8'), file).toMatch(/isEnterprise/)
    }
  })
})
