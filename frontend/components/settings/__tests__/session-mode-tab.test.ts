import { describe, expect, it } from 'vitest'
import { PROJECTS_ENV_LINES, describeProjectsFolder, type SessionModeSettings } from '@/components/settings/SessionModeTab'

const base: SessionModeSettings = {
  default_folder: 'projects',
  default_folder_explicit: false,
  local_projects_dir: '/Users/me/Development',
  projects_mount: 'rw',
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

  it('hands the operator the two .env lines for their folder', () => {
    expect(PROJECTS_ENV_LINES('/Users/me/Development')).toBe('LOCAL_PROJECTS_DIR=/Users/me/Development\nLOCAL_PROJECTS_MOUNT=rw')
  })
})
