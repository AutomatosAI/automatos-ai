/**
 * #942 — the "Session tools" picker of a CLI agent, and the workspace gaps shown before a switch.
 *
 * F329 (4 Oct): a Business Analyst moved to a CLI session could not query the shop database and
 * nothing on the agent page had said so. The page now reads `session_tool_groups` and
 * `session_tool_gaps` from `GET /api/agents/{id}`, previews a new selection with `?groups=`, and
 * saves `configuration.session_tool_groups` only when the owner changed the boxes.
 */
import { useState } from 'react'
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'

const AGENT_ID = 7
const ALL_GROUPS = ['data', 'graph', 'documents', 'playbooks', 'reports', 'missions']

const GROUPS = {
  enabled: ALL_GROUPS,
  is_default: true,
  available: [
    { id: 'data', label: 'Data', description: "Query the workspace's connected databases.", tools: ['query_database'] },
    { id: 'graph', label: 'Knowledge Graph', description: 'Ask the Knowledge Graph.', tools: ['query_graph'] },
    { id: 'documents', label: 'Documents', description: 'Make PDF and Word files.', tools: ['generate_document'] },
    { id: 'playbooks', label: 'Playbooks', description: 'List and run playbooks.', tools: ['list_playbooks', 'run_playbook'] },
    { id: 'reports', label: 'Reports', description: "Read another agent's latest report.", tools: ['get_latest_report'] },
    { id: 'missions', label: 'Missions', description: 'Mission tools.', tools: ['mission_status'] },
  ],
}

const SKILL_GAP = { kind: 'skill', skill: 'web-research', tools: [], instead: { platform_submit_report: 'submit_report' } }
const DATABASE_GAP = {
  kind: 'workspace',
  capability: 'database',
  group: 'data',
  message: "This workspace has a connected database, but this agent's sessions can't query it.",
  fix: { enable_group: 'data' },
}

// The settings list may name every session tool; the picker's "always on" line drops the grouped ones.
const SETTINGS = {
  session_tools: [
    { name: 'board_summary', description: '' },
    { name: 'search_knowledge', description: '' },
    { name: 'query_database', description: '' },
  ],
}

type Detail = Record<string, unknown>

/** apiClient.request by path: the saved agent, its `?groups=` previews, and the session-mode endpoints. */
function apiMock(saved: Detail, previews: Record<string, Detail> = {}) {
  return vi.fn(async (path: string) => {
    if (path === '/api/v1/cli-hosts/health') return { registry: [], providers_online: [] }
    if (path === '/api/v1/cli-hosts/settings') return SETTINGS
    if (path === `/api/agents/${AGENT_ID}`) return saved
    const groups = path.split('?groups=')[1]
    if (groups !== undefined) return previews[groups] ?? { ...saved, session_tool_gaps: [] }
    throw new Error(`unexpected path ${path}`)
  })
}

async function load(request: ReturnType<typeof apiMock>) {
  vi.resetModules()
  vi.doMock('@/lib/auth-edition', () => ({ authEdition: 'local', isLocal: true, isSaaS: false }))
  vi.doMock('@/lib/api-client', () => ({ apiClient: { request } }))
  const runtime = await import('../runtime-section')
  const { ConfiguredAgentContext } = await import('../configured-agent-context')
  return { ...runtime, ConfiguredAgentContext }
}

afterEach(() => {
  vi.doUnmock('@/lib/auth-edition')
  vi.doUnmock('@/lib/api-client')
  vi.resetModules()
})

type Loaded = Awaited<ReturnType<typeof load>>

/**
 * The modal's own wiring, in miniature: form state seeded from the saved configuration, every
 * edit through `onChange(field, value)`, and the save fragment from
 * `runtimeConfiguration(normalizeRuntimeFields(form))` — exactly what handleSave sends.
 */
function renderForm(mod: Loaded, configuration: Record<string, unknown>, savedGaps: unknown[] | null = null) {
  const onSave = vi.fn()
  function Form() {
    const [form, setForm] = useState<Record<string, unknown>>(() => ({ ...mod.runtimeFieldsFromConfiguration(configuration) }))
    return (
      <mod.ConfiguredAgentContext.Provider value={AGENT_ID}>
        <mod.RuntimeSection
          value={mod.normalizeRuntimeFields(form)}
          onChange={(field, value) => setForm((prev) => ({ ...prev, [field]: value }))}
          sessionToolGaps={savedGaps as never}
        />
        <button type="button" onClick={() => onSave(mod.runtimeConfiguration(mod.normalizeRuntimeFields(form)))}>
          save
        </button>
      </mod.ConfiguredAgentContext.Provider>
    )
  }
  render(<Form />)
  return onSave
}

const CLI_AGENT = { runtime: 'cli', provider: 'claude', working_directory: '/w' }

describe('Session tools picker', () => {
  it('lists every group with its description and tools, the core tools as always on, and the default note', async () => {
    const mod = await load(apiMock({ session_tool_groups: GROUPS, session_tool_gaps: [] }))
    renderForm(mod, CLI_AGENT, [])
    const picker = await screen.findByTestId('session-tool-groups')
    for (const group of GROUPS.available) {
      expect(screen.getByRole('checkbox', { name: group.label })).toBeChecked()
      expect(picker.textContent).toContain(group.description)
    }
    expect(picker.textContent).toContain('list_playbooks')
    expect(screen.getByTestId('session-tool-groups-default')).toHaveTextContent('Defaults: all on')
    const alwaysOn = await screen.findByText(/Always on:/)
    // read from the mocked settings, not the SESSION_CORE_TOOLS fallback (which also names ask_human):
    // before the shared lazy import, this call reached the real client and the fallback hid it
    await waitFor(() => expect(alwaysOn.textContent).not.toContain('ask_human'))
    expect(alwaysOn.textContent).toContain('board_summary')
    expect(alwaysOn.textContent).toContain('search_knowledge')
    expect(alwaysOn.textContent).not.toContain('query_database')
    // the old flat tool line gives way to the picker
    expect(screen.queryByTestId('session-tools')).not.toBeInTheDocument()
  })

  it('unticking a group clears the default note and previews the gaps of the new selection', async () => {
    const request = apiMock({ session_tool_groups: GROUPS, session_tool_gaps: [] })
    const mod = await load(request)
    renderForm(mod, CLI_AGENT, [])
    fireEvent.click(await screen.findByRole('checkbox', { name: 'Knowledge Graph' }))
    expect(screen.getByRole('checkbox', { name: 'Knowledge Graph' })).not.toBeChecked()
    expect(screen.queryByTestId('session-tool-groups-default')).not.toBeInTheDocument()
    await waitFor(
      () => expect(request).toHaveBeenCalledWith(`/api/agents/${AGENT_ID}?groups=data,documents,playbooks,reports,missions`),
      { timeout: 2000 },
    )
  })
})

describe('Workspace gaps before and during a switch to cli', () => {
  it('warns about a connected database the selection cannot reach, and "Turn on Data" ticks the group', async () => {
    const saved = { session_tool_groups: { ...GROUPS, enabled: ['graph'], is_default: false }, session_tool_gaps: [DATABASE_GAP, SKILL_GAP] }
    const mod = await load(apiMock(saved, { graph: saved }))
    const onSave = renderForm(mod, { ...CLI_AGENT, session_tool_groups: ['graph'] }, [DATABASE_GAP, SKILL_GAP])
    const warning = await screen.findByTestId('session-capability-gaps')
    expect(warning).toHaveTextContent("This workspace has a connected database, but this agent's sessions can't query it.")
    // the skill gaps still read as before, and the workspace entry adds no line of its own there
    const skillLines = screen.getByTestId('session-tool-gaps')
    expect(skillLines.textContent).toContain('web-research calls `platform_submit_report`')
    expect(skillLines.querySelectorAll('p')).toHaveLength(1)
    await waitFor(() => expect(screen.getByRole('checkbox', { name: 'Data' })).not.toBeChecked())
    fireEvent.click(screen.getByRole('button', { name: 'Turn on Data' }))
    expect(screen.getByRole('checkbox', { name: 'Data' })).toBeChecked()
    expect(screen.queryByRole('button', { name: 'Turn on Data' })).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'save' }))
    expect(onSave).toHaveBeenCalledWith(expect.objectContaining({ session_tool_groups: ['data', 'graph'] }))
  })

  it('asks for the gaps of a saved api agent being switched to cli (its own detail carries none)', async () => {
    const preview = { session_tool_groups: GROUPS, session_tool_gaps: [DATABASE_GAP] }
    const request = apiMock({ session_tool_groups: GROUPS, session_tool_gaps: null }, { [ALL_GROUPS.join(',')]: preview })
    const mod = await load(request)
    renderForm(mod, CLI_AGENT, null)
    expect(await screen.findByTestId('session-capability-gaps', {}, { timeout: 2000 })).toHaveTextContent('connected database')
    expect(request).toHaveBeenCalledWith(`/api/agents/${AGENT_ID}?groups=${ALL_GROUPS.join(',')}`)
  })
})

describe('Saving the session tool groups', () => {
  it('sends nothing for an untouched agent, so it keeps "absent = every group"', async () => {
    const mod = await load(apiMock({ session_tool_groups: GROUPS, session_tool_gaps: [] }))
    const onSave = renderForm(mod, CLI_AGENT, [])
    await screen.findByTestId('session-tool-groups')
    fireEvent.click(screen.getByRole('button', { name: 'save' }))
    expect(onSave.mock.calls[0][0]).not.toHaveProperty('session_tool_groups')
  })

  it('sends the ticked groups in display order once the owner changed a box', async () => {
    const mod = await load(apiMock({ session_tool_groups: GROUPS, session_tool_gaps: [] }))
    const onSave = renderForm(mod, CLI_AGENT, [])
    fireEvent.click(await screen.findByRole('checkbox', { name: 'Documents' }))
    fireEvent.click(screen.getByRole('checkbox', { name: 'Missions' }))
    fireEvent.click(screen.getByRole('button', { name: 'save' }))
    expect(onSave.mock.calls[0][0].session_tool_groups).toEqual(['data', 'graph', 'playbooks', 'reports'])
  })

  it('an api agent never carries the field, even after an edit', async () => {
    const mod = await load(apiMock({}))
    expect(mod.runtimeConfiguration({ ...mod.DEFAULT_RUNTIME_FIELDS, cli_session_tool_groups: ['data'] })).toEqual({ runtime: 'api' })
  })
})

describe('An older backend without session_tool_groups', () => {
  it('hides the picker and keeps the flat tool line', async () => {
    const request = apiMock({ session_tool_gaps: [] })
    const mod = await load(request)
    renderForm(mod, CLI_AGENT, [])
    expect(await screen.findByTestId('session-tools')).toHaveTextContent('board_summary')
    await waitFor(() => expect(request).toHaveBeenCalledWith(`/api/agents/${AGENT_ID}`))
    expect(screen.queryByTestId('session-tool-groups')).not.toBeInTheDocument()
    expect(request.mock.calls.some(([path]) => String(path).includes('?groups='))).toBe(false)
  })

  it('outside the configuration modal (the create wizard) there is no agent to read, so no picker', async () => {
    const request = apiMock({ session_tool_groups: GROUPS })
    const mod = await load(request)
    render(<mod.RuntimeSection value={mod.normalizeRuntimeFields(CLI_AGENT)} onChange={vi.fn()} />)
    await screen.findByTestId('session-tools')
    expect(screen.queryByTestId('session-tool-groups')).not.toBeInTheDocument()
    expect(request.mock.calls.some(([path]) => String(path).startsWith('/api/agents/'))).toBe(false)
  })
})

// Two hooks of one module importing the mocked client in the same tick raced in vitest: the second
// got the real client (CI, 4 Oct). Every runtime hook now shares one lazy import.
describe('loadApiClient', () => {
  it('hands concurrent callers the same (mocked) client from one import', async () => {
    const request = apiMock({})
    await load(request)
    const { loadApiClient } = await import('../lazy-api-client')
    const [first, second] = await Promise.all([loadApiClient(), loadApiClient()])
    expect(first).toBe(second)
    expect(first.request).toBe(request)
  })
})

describe('session tool group helpers', () => {
  it('reads nothing from a missing or malformed field, and drops unknown enabled ids', async () => {
    const { normalizeSessionToolGroups } = await import('../session-tool-groups-model')
    expect(normalizeSessionToolGroups(undefined)).toBeNull()
    expect(normalizeSessionToolGroups({ enabled: [], available: [] })).toBeNull()
    expect(normalizeSessionToolGroups({ enabled: ['data', 'bogus'], is_default: false, available: [{ id: 'data' }] })).toEqual({
      enabled: ['data'],
      is_default: false,
      available: [{ id: 'data', label: 'data', description: '', tools: [] }],
    })
  })

  it('keeps the display order whatever order the boxes are ticked in', async () => {
    const { normalizeSessionToolGroups, toggleGroup } = await import('../session-tool-groups-model')
    const groups = normalizeSessionToolGroups(GROUPS)!
    expect(toggleGroup(groups, ['missions', 'graph'], 'data', true)).toEqual(['data', 'graph', 'missions'])
    expect(toggleGroup(groups, ['data', 'graph'], 'data', false)).toEqual(['graph'])
  })

  it('builds the preview path, an empty selection included', async () => {
    const { agentDetailPath } = await import('../session-tool-groups-model')
    expect(agentDetailPath(7, null)).toBe('/api/agents/7')
    expect(agentDetailPath(7, ['data', 'graph'])).toBe('/api/agents/7?groups=data,graph')
    expect(agentDetailPath(7, [])).toBe('/api/agents/7?groups=')
  })
})
