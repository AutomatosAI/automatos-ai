/**
 * The marketplace's admin controls (Import from GitHub, approve / reject) used
 * to show only for a Clerk user whose email contained "automatos.app". A local
 * install has no Clerk user, so the local operator never saw Import from GitHub
 * although the backend already accepts them (the operator is super_admin).
 * The gate is the role context now — the same system_role hierarchy the admin
 * routes enforce. This guard keeps the email check from creeping back.
 */
import { describe, it, expect } from 'vitest'
import { readFileSync } from 'fs'
import path from 'path'

const TABS = [
  'marketplace-plugins-tab.tsx',
  'marketplace-agents-tab.tsx',
  'marketplace-playbooks-tab.tsx',
]

describe('marketplace admin gate', () => {
  for (const tab of TABS) {
    it(`${tab} gates on the system role, not an email domain`, () => {
      const src = readFileSync(path.resolve(__dirname, '..', tab), 'utf8')
      expect(src).not.toContain('automatos.app')
      expect(src).not.toContain('emailAddresses')
      expect(src).toContain("useSystemRole } from '@/contexts/role-context'")
      expect(src).toContain('const { isAdmin } = useSystemRole()')
    })
  }
})

describe('skills tab empty state', () => {
  it('points a fresh install at the baseline library and opens the import on it', () => {
    const src = readFileSync(path.resolve(__dirname, '..', 'marketplace-skills-tab.tsx'), 'utf8')
    expect(src).not.toContain('Plugins > Import from GitHub')
    expect(src).toContain('BASELINE_SKILLS_REPO_LABEL')
    expect(src).toContain('initialUrl={BASELINE_SKILLS_REPO_URL}')
    expect(src).toContain('const { isAdmin } = useSystemRole()')
  })
})
