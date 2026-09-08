import { describe, it, expect } from 'vitest'
import { normalizeCodeRoot, canvasTitleFor, WORKSPACE_ROOT } from '../code-root'
import { repoRootOptions } from '../useRepoRoots'

describe('PRD-235 W2 — code roots', () => {
  it('normalizes workspace-relative folders and refuses escapes', () => {
    expect(normalizeCodeRoot(undefined)).toBeNull()
    expect(normalizeCodeRoot('')).toBe(WORKSPACE_ROOT)
    expect(normalizeCodeRoot('.')).toBe(WORKSPACE_ROOT)
    expect(normalizeCodeRoot('./sessions/71/')).toBe('sessions/71')
    expect(normalizeCodeRoot('projects/my-repo')).toBe('projects/my-repo')
    expect(normalizeCodeRoot('/etc')).toBeNull()
    expect(normalizeCodeRoot('projects/../secret')).toBeNull()
  })

  it('titles the Canvas by its root', () => {
    expect(canvasTitleFor(WORKSPACE_ROOT)).toBe('Code Canvas')
    expect(canvasTitleFor('projects/my-repo')).toBe('Code Canvas · projects/my-repo')
  })

  it('offers the workspace root, repos under projects/ and the newest sessions first', () => {
    const opts = repoRootOptions(['projects/repo-a'], ['sessions/68', 'sessions/71', 'sessions/70'])
    expect(opts[0]).toEqual({ value: '.', label: 'Workspace root', group: 'workspace' })
    expect(opts[1]).toEqual({ value: 'projects/repo-a', label: 'repo-a', group: 'projects' })
    expect(opts.slice(2).map((o) => o.value)).toEqual(['sessions/71', 'sessions/70', 'sessions/68'])
  })
})
