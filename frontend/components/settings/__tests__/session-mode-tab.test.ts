import { describe, expect, it } from 'vitest'
import {
  PROJECTS_ENV_LINES,
  describeDeliverablesRoot,
  describeProjectsFolder,
  suggestedDeliverablesRoot,
  type SessionModeSettings,
} from '@/components/settings/SessionModeTab'

const base: SessionModeSettings = {
  default_folder: 'projects',
  default_folder_explicit: false,
  local_projects_dir: '/Users/me/Development',
  projects_mount: 'rw',
  workspace_dir: '/Users/me/Development/deliverables',
  host_allowed_roots: ['/Users/me/ws/workspaces', '/Users/me/Development'],
}

describe('Settings → Session mode (PRD-239 S6c)', () => {
  it('says the projects folder is mounted, writable and allowed when it is', () => {
    const s = describeProjectsFolder(base)
    expect(s.tone).toBe('ok')
    expect(s.text).toContain('/Users/me/Development')
    expect(s.text).toContain('can save into it')
  })

  it('flags a folder the host does not allow yet, and a read-only mount', () => {
    const s = describeProjectsFolder({ ...base, host_allowed_roots: ['/Users/me/ws/workspaces'], projects_mount: 'ro' })
    expect(s.tone).toBe('warn')
    expect(s.text).toContain('make cli-host-install')
    expect(s.text).toContain('read-only')
  })

  it('accepts a sub-folder of an allowed root and says so when nothing is set', () => {
    expect(describeProjectsFolder({ ...base, host_allowed_roots: ['/Users/me'] }).tone).toBe('ok')
    expect(describeProjectsFolder({ ...base, local_projects_dir: null }).tone).toBe('warn')
    expect(describeProjectsFolder(undefined).tone).toBe('muted')
  })

  it('hands the operator the .env lines for their folders', () => {
    expect(PROJECTS_ENV_LINES('/Users/me/Development')).toBe('LOCAL_PROJECTS_DIR=/Users/me/Development\nLOCAL_PROJECTS_MOUNT=rw')
    expect(PROJECTS_ENV_LINES('/Users/me/Development', '/Users/me/Development/deliverables')).toBe(
      'LOCAL_PROJECTS_DIR=/Users/me/Development\nLOCAL_PROJECTS_MOUNT=rw\nAUTOMATOS_WORKSPACE_DIR=/Users/me/Development/deliverables',
    )
    expect(suggestedDeliverablesRoot('/Users/me/Development/')).toBe('/Users/me/Development/deliverables')
  })
})

describe('Settings → Session mode — the deliverables root', () => {
  it('is green when mounted and allowed by the host (a sub-folder of an allowed root counts)', () => {
    const s = describeDeliverablesRoot(base)
    expect(s.tone).toBe('ok')
    expect(s.text).toContain('/Users/me/Development/deliverables')
    expect(s.text).toContain('sessions/<ticket>')
  })

  it('warns when the host does not allow it yet', () => {
    const s = describeDeliverablesRoot({ ...base, workspace_dir: '/Users/me/Automatos/deliverables' })
    expect(s.tone).toBe('warn')
    expect(s.text).toContain('make cli-host-install')
  })

  it('explains the compose default when the stack did not export the root', () => {
    const s = describeDeliverablesRoot({ ...base, workspace_dir: null })
    expect(s.tone).toBe('muted')
    expect(s.text).toContain('AUTOMATOS_WORKSPACE_DIR')
    expect(describeDeliverablesRoot(undefined).tone).toBe('muted')
  })
})
