/**
 * PRD-234 S4 — the Runtime group shared by the create wizard and Configure.
 *
 * Local edition only; the fields round-trip through `Agent.configuration`
 * (runtime / provider / model / working_directory) with blanks saved as null.
 */
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'

const HEALTH = {
  registry: [
    { id: 'claude', label: 'Claude Code', model_hint: 'aliases', model_placeholder: 'fable · opus' },
    { id: 'codex', label: 'Codex', model_hint: 'a ChatGPT-plan model', model_placeholder: 'gpt-5.5' },
  ],
  providers_online: ['claude'],
}

// PRD-245 S1.5: `GET /api/v1/cli-hosts/settings` → `session_tools`, in the order a session is offered them.
const SESSION_TOOLS = [
  { name: 'board_summary', description: "Counts of this workspace's board tasks by status." },
  { name: 'list_tasks', description: 'The board tasks of this workspace, newest first.' },
  { name: 'update_ticket', description: 'Move this ticket and add a note; a session never closes it.' },
  { name: 'submit_report', description: 'Attach a report to this ticket.' },
  { name: 'search_knowledge', description: "Search this workspace's knowledge." },
]

async function load(edition: 'local' | 'saas') {
  vi.resetModules()
  vi.doMock('@/lib/auth-edition', () => ({
    authEdition: edition,
    isLocal: edition === 'local',
    isSaaS: edition === 'saas',
  }))
  // CLI adapter design §8.3: the picker asks the backend which CLIs exist and which are served.
  vi.doMock('@/lib/api-client', () => ({ apiClient: { request: vi.fn().mockResolvedValue(HEALTH) } }))
  return import('../runtime-section')
}

afterEach(() => {
  vi.doUnmock('@/lib/auth-edition')
  vi.doUnmock('@/lib/api-client')
  vi.resetModules()
})

// CLI adapter design §8.3: the options come from the registry; a CLI no online host runs says so;
// a saved choice survives an unreadable registry.
describe('providerOptions', () => {
  it('lists every registry CLI and notes the ones no online host runs', async () => {
    const { providerOptions } = await load('local')
    const options = providerOptions(HEALTH, 'claude')
    expect(options.map((o) => o.id)).toEqual(['claude', 'codex'])
    expect(options[0]).toMatchObject({ label: 'Claude Code', served: true, note: null })
    expect(options[1]).toMatchObject({ label: 'Codex', served: false, note: 'no online host runs it yet' })
  })

  it('keeps the current value when the registry is unavailable or does not know it', async () => {
    const { providerOptions } = await load('local')
    expect(providerOptions(null, 'codex')).toEqual([{ id: 'codex', label: 'codex', served: false, note: null }])
    const unknown = providerOptions(HEALTH, 'grok')
    expect(unknown.map((o) => o.id)).toEqual(['claude', 'codex', 'grok'])
    expect(unknown[2].note).toContain('not in this instance')
  })
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

// PRD-245 S1.5: the Automatos tools a ticket session may call, as the form says them.
// The list rides `GET /api/v1/cli-hosts/settings`; an older backend carries no
// field at all and the line must simply not render.
describe('normalizeSessionTools', () => {
  it('renders nothing when the backend does not carry the list', async () => {
    const { normalizeSessionTools } = await load('local')
    expect(normalizeSessionTools(undefined)).toEqual([])
    expect(normalizeSessionTools(null)).toEqual([])
    expect(normalizeSessionTools([])).toEqual([])
    expect(normalizeSessionTools('board_summary')).toEqual([])
  })

  it('keeps the wave-1 tools in the order the API returned them, with their descriptions', async () => {
    const { normalizeSessionTools } = await load('local')
    const tools = normalizeSessionTools(SESSION_TOOLS)
    expect(tools.map((t) => t.name)).toEqual([
      'board_summary',
      'list_tasks',
      'update_ticket',
      'submit_report',
      'search_knowledge',
    ])
    expect(tools[0].description).toContain('board tasks by status')
  })

  it('drops a row with no name and accepts one with no description', async () => {
    const { normalizeSessionTools } = await load('local')
    expect(normalizeSessionTools([{ name: '  ' }, { description: 'orphan' }, null, { name: ' board_summary ' }])).toEqual([
      { name: 'board_summary', description: '' },
    ])
  })
})

// PRD-245 S1.5: one line per skill of the agent whose body calls tools by the API
// agents' names — what the same work is called in a session, and what a session
// cannot do at all. Help text, never an error: the agent is not broken.
describe('describeSessionToolGap', () => {
  it('says nothing when the entry has nothing to say', async () => {
    const { describeSessionToolGap } = await load('local')
    expect(describeSessionToolGap(undefined)).toBe('')
    expect(describeSessionToolGap(null)).toBe('')
    expect(describeSessionToolGap({ skill: 'web-research', tools: [] })).toBe('')
    expect(describeSessionToolGap({ skill: 'web-research', tools: [], instead: {} })).toBe('')
  })

  it('names the session tool that does the same job', async () => {
    const { describeSessionToolGap } = await load('local')
    expect(
      describeSessionToolGap({ skill: 'web-research', tools: [], instead: { platform_submit_report: 'submit_report' } }),
    ).toBe('web-research calls `platform_submit_report` — in a session that work is `submit_report`.')
  })

  it('names the tools a session cannot use at all', async () => {
    const { describeSessionToolGap } = await load('local')
    expect(describeSessionToolGap({ skill: 'web-research', tools: ['composio_execute'] })).toBe(
      'web-research calls `composio_execute`, which a session cannot use.',
    )
    expect(describeSessionToolGap({ skill: 'web-research', tools: ['composio_execute', 'scratchpad_write'] })).toBe(
      'web-research calls `composio_execute` and `scratchpad_write`, which a session cannot use.',
    )
  })

  it('combines both shapes for one skill into one line, replacements first', async () => {
    const { describeSessionToolGap } = await load('local')
    expect(
      describeSessionToolGap({
        skill: 'web-research',
        tools: ['composio_execute'],
        instead: { platform_submit_report: 'submit_report', platform_board_summary: 'board_summary' },
      }),
    ).toBe(
      'web-research calls `platform_submit_report` and `platform_board_summary` — in a session that work is ' +
        '`submit_report` and `board_summary`. It also calls `composio_execute`, which a session cannot use.',
    )
  })

  it('survives a gap with no skill name and ignores half-written pairs', async () => {
    const { describeSessionToolGap } = await load('local')
    expect(describeSessionToolGap({ skill: '', tools: ['composio_execute'] })).toBe(
      'A skill calls `composio_execute`, which a session cannot use.',
    )
    expect(describeSessionToolGap({ skill: 'notes', tools: null, instead: { platform_submit_report: '' } })).toBe('')
  })
})

// PRD-245 S1.5: the gap lines come from the agent detail, not the form, so the
// caller that fetched the agent passes them; the create wizard passes nothing.
describe('RuntimeSection session tool gaps', () => {
  const cliFields = {
    runtime: 'cli' as const,
    cli_provider: 'claude',
    cli_model: '',
    cli_working_directory: '',
    cli_worktree: true,
  }

  it('reads one line per skill that calls tools a session works differently on', async () => {
    const { RuntimeSection } = await load('local')
    render(
      <RuntimeSection
        value={cliFields}
        onChange={vi.fn()}
        sessionToolGaps={[
          { skill: 'web-research', tools: ['composio_execute'], instead: { platform_submit_report: 'submit_report' } },
          { skill: 'note-taking', tools: [], instead: { platform_board_summary: 'board_summary' } },
        ]}
      />,
    )
    const gaps = screen.getByTestId('session-tool-gaps')
    expect(gaps.textContent).toContain('web-research calls `platform_submit_report` — in a session that work is `submit_report`.')
    expect(gaps.textContent).toContain('It also calls `composio_execute`, which a session cannot use.')
    expect(gaps.textContent).toContain('note-taking calls `platform_board_summary`')
  })

  it('says nothing for an api-runtime agent (null), an agent with no gaps ([]), or an older backend', async () => {
    const { RuntimeSection } = await load('local')
    const { rerender } = render(<RuntimeSection value={cliFields} onChange={vi.fn()} sessionToolGaps={null} />)
    expect(screen.queryByTestId('session-tool-gaps')).not.toBeInTheDocument()
    // the settings response of an older backend carries no `session_tools` at all
    expect(screen.queryByTestId('session-tools')).not.toBeInTheDocument()
    rerender(<RuntimeSection value={cliFields} onChange={vi.fn()} sessionToolGaps={[]} />)
    expect(screen.queryByTestId('session-tool-gaps')).not.toBeInTheDocument()
    rerender(<RuntimeSection value={cliFields} onChange={vi.fn()} />)
    expect(screen.queryByTestId('session-tool-gaps')).not.toBeInTheDocument()
  })
})
