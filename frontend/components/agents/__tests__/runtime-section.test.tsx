/**
 * PRD-234 S4 — the Runtime group shared by the create wizard and Configure.
 *
 * Local edition only; the fields round-trip through `Agent.configuration`
 * (runtime / provider / model / working_directory) with blanks saved as null.
 */
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'

async function load(edition: 'local' | 'saas') {
  vi.resetModules()
  vi.doMock('@/lib/auth-edition', () => ({
    authEdition: edition,
    isLocal: edition === 'local',
    isSaaS: edition === 'saas',
  }))
  return import('../runtime-section')
}

afterEach(() => {
  vi.doUnmock('@/lib/auth-edition')
  vi.resetModules()
})

describe('runtime configuration helpers', () => {
  it('an api agent saves only the runtime kind', async () => {
    const { runtimeConfiguration, DEFAULT_RUNTIME_FIELDS } = await load('local')
    expect(runtimeConfiguration(DEFAULT_RUNTIME_FIELDS)).toEqual({ runtime: 'api' })
  })

  it('a cli agent saves provider/model/working_directory with blanks as null, trimmed', async () => {
    const { runtimeConfiguration } = await load('local')
    expect(
      runtimeConfiguration({ runtime: 'cli', cli_provider: 'claude', cli_model: '  ', cli_working_directory: '', cli_worktree: true }),
    ).toEqual({ runtime: 'cli', provider: 'claude', model: null, working_directory: null, worktree_per_ticket: true })
    expect(
      runtimeConfiguration({ runtime: 'cli', cli_provider: '', cli_model: ' fable ', cli_working_directory: ' /w/repo ', cli_worktree: true }),
    ).toEqual({ runtime: 'cli', provider: 'claude', model: 'fable', working_directory: '/w/repo', worktree_per_ticket: true })
  })

  it('reads the fields back from an agent configuration, defaulting anything missing', async () => {
    const { runtimeFieldsFromConfiguration, DEFAULT_RUNTIME_FIELDS } = await load('local')
    expect(runtimeFieldsFromConfiguration(undefined)).toEqual(DEFAULT_RUNTIME_FIELDS)
    expect(runtimeFieldsFromConfiguration({ runtime: 'api', model: 'gpt-4o' })).toEqual({
      ...DEFAULT_RUNTIME_FIELDS,
      cli_model: 'gpt-4o',
      cli_worktree: true,
    })
    expect(
      runtimeFieldsFromConfiguration({ runtime: 'cli', provider: 'claude', model: 'opus', working_directory: '/w' }),
    ).toEqual({ runtime: 'cli', cli_provider: 'claude', cli_model: 'opus', cli_working_directory: '/w', cli_worktree: true })
  })
})

describe('RuntimeSection', () => {
  it('does not exist in the saas edition', async () => {
    const { RuntimeSection, DEFAULT_RUNTIME_FIELDS } = await load('saas')
    const { container } = render(<RuntimeSection value={DEFAULT_RUNTIME_FIELDS} onChange={vi.fn()} />)
    expect(container).toBeEmptyDOMElement()
  })

  it('renders the runtime choice locally and hides the session fields for an api agent', async () => {
    const { RuntimeSection, DEFAULT_RUNTIME_FIELDS } = await load('local')
    render(<RuntimeSection value={DEFAULT_RUNTIME_FIELDS} onChange={vi.fn()} />)
    expect(screen.getByLabelText('Runtime')).toBeInTheDocument()
    expect(screen.queryByLabelText(/Workspace folder/)).not.toBeInTheDocument()
  })

  it('shows the session fields for a cli agent and reports edits field by field', async () => {
    const { RuntimeSection } = await load('local')
    const onChange = vi.fn()
    render(
      <RuntimeSection
        value={{ runtime: 'cli', cli_provider: 'claude', cli_model: '', cli_working_directory: '' }}
        onChange={onChange}
      />,
    )
    fireEvent.change(screen.getByLabelText(/Model \(optional\)/), { target: { value: 'fable' } })
    fireEvent.change(screen.getByLabelText(/Workspace folder/), { target: { value: '/w/repo' } })
    expect(onChange).toHaveBeenCalledWith('cli_model', 'fable')
    expect(onChange).toHaveBeenCalledWith('cli_working_directory', '/w/repo')
  })
})

// PRD-239 S6: the verdict line the operator reads under the working directory.
describe('describeWorkspaceCheck', () => {
  const base = {
    path: '/Users/me/Development/repo',
    valid: true,
    errors: [] as string[],
    explorer_root: 'projects/repo' as string | null,
    browsable: true,
    allowed: true as boolean | null,
    allowed_roots: ['/Users/me/Development'],
    projects_dir: '/Users/me/Development' as string | null,
  }

  it('names the Canvas root when the folder is browsable and allowed', async () => {
    const { describeWorkspaceCheck } = await load('local')
    const verdict = describeWorkspaceCheck(base)
    expect(verdict.tone).toBe('ok')
    expect(verdict.text).toContain('projects/repo')
    expect(verdict.canvasRoot).toBe('projects/repo')
  })

  it('says when no host could confirm the folder', async () => {
    const { describeWorkspaceCheck } = await load('local')
    const verdict = describeWorkspaceCheck({ ...base, allowed: null, allowed_roots: [] })
    expect(verdict.tone).toBe('ok')
    expect(verdict.text).toContain('no host online')
  })

  it('refuses a folder outside the host allow-list before anything else', async () => {
    const { describeWorkspaceCheck } = await load('local')
    const verdict = describeWorkspaceCheck({ ...base, allowed: false, allowed_roots: ['/Users/me/ws'] })
    expect(verdict.tone).toBe('error')
    expect(verdict.text).toContain('/Users/me/ws')
    expect(verdict.canvasRoot).toBeNull()
  })

  it('warns when sessions can run but the folder is not browsable', async () => {
    const { describeWorkspaceCheck } = await load('local')
    const verdict = describeWorkspaceCheck({ ...base, explorer_root: null, browsable: false })
    expect(verdict.tone).toBe('warn')
    expect(verdict.text).toContain('not browsable')
    expect(describeWorkspaceCheck({ ...base, explorer_root: null, browsable: false, projects_dir: null }).text).toContain(
      'LOCAL_PROJECTS_DIR is not set',
    )
  })

  it('surfaces the validation error verbatim', async () => {
    const { describeWorkspaceCheck } = await load('local')
    const verdict = describeWorkspaceCheck({ ...base, valid: false, errors: ['must be an absolute path'] })
    expect(verdict.tone).toBe('error')
    expect(verdict.text).toBe('must be an absolute path')
  })
})

describe('worktree per ticket (PRD-239)', () => {
  it('round-trips the agent choice and defaults to on', async () => {
    const { runtimeConfiguration, runtimeFieldsFromConfiguration } = await import('@/components/agents/runtime-section')
    const off = runtimeConfiguration({ runtime: 'cli', cli_provider: 'claude', cli_model: '', cli_working_directory: '/Users/me/Development', cli_worktree: false })
    expect(off.worktree_per_ticket).toBe(false)
    expect(runtimeFieldsFromConfiguration({ runtime: 'cli', provider: 'claude', working_directory: '/w', worktree_per_ticket: false }).cli_worktree).toBe(false)
    expect(runtimeFieldsFromConfiguration({ runtime: 'cli', provider: 'claude', working_directory: '/w' }).cli_worktree).toBe(true)
  })
})
