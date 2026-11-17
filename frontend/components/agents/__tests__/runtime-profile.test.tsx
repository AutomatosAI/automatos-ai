/**
 * PRD-274 — the "Runtime profile" choice (Full / Lean) of a saved agent, on the Model tab of its
 * configuration modal, under the Runtime group.
 *
 * An absent or `full` profile shows Full. The pick rides the form, so the modal's existing save
 * sends `configuration.runtime_profile` only when the person chose one, beside everything it
 * already sent. A system agent is never lean (the backend says so), so its choice is fixed at
 * Full and cannot be saved. The Runtime group above is local only; the profile is not.
 */
import { useState } from 'react'
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'

const AGENT_ID = 7

type Detail = Record<string, unknown>

/** apiClient.request by path: the saved agent (or its failure), and the session-mode reads of a cli agent. */
function apiMock(saved: Detail | Error) {
  return vi.fn(async (path: string) => {
    if (path === '/api/v1/cli-hosts/health') return { registry: [], providers_online: [] }
    if (path === '/api/v1/cli-hosts/settings') return { session_tools: [] }
    if (path === `/api/agents/${AGENT_ID}`) {
      if (saved instanceof Error) throw saved
      return saved
    }
    throw new Error(`unexpected path ${path}`)
  })
}

async function load(edition: 'local' | 'saas', request: ReturnType<typeof apiMock>) {
  vi.resetModules()
  vi.doMock('@/lib/auth-edition', () => ({
    authEdition: edition,
    isLocal: edition === 'local',
    isSaaS: edition === 'saas',
  }))
  vi.doMock('@/lib/api-client', () => ({ apiClient: { request } }))
  const runtime = await import('../runtime-section')
  const { ConfiguredAgentContext } = await import('../configured-agent-context')
  return { ...runtime, ConfiguredAgentContext }
}

afterEach(() => {
  vi.doUnmock('@/lib/auth-edition')
  vi.doUnmock('@/lib/api-client')
  vi.resetModules()
  vi.restoreAllMocks()
})

type Loaded = Awaited<ReturnType<typeof load>>

/**
 * The modal's own wiring, in miniature (as in session-tool-groups.test.tsx): form state seeded
 * from the saved configuration, every edit through `onChange(field, value)`, and the save
 * fragment from `runtimeConfiguration(normalizeRuntimeFields(form))`, which is what handleSave
 * spreads into the `configuration` it sends with `PUT /api/agents/{id}`.
 */
function renderForm(mod: Loaded, configuration: Detail) {
  const onSave = vi.fn()
  function Form() {
    const [form, setForm] = useState<Detail>(() => ({ ...mod.runtimeFieldsFromConfiguration(configuration) }))
    return (
      <mod.ConfiguredAgentContext.Provider value={AGENT_ID}>
        <mod.RuntimeSection
          value={mod.normalizeRuntimeFields(form)}
          onChange={(field, value) => setForm((prev) => ({ ...prev, [field]: value }))}
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

/** What the save sent before this change for the same configuration: the baseline a pick adds to. */
function fragmentBefore(mod: Loaded, configuration: Detail) {
  return mod.runtimeConfiguration(mod.runtimeFieldsFromConfiguration(configuration))
}

const radio = (name: 'Full' | 'Lean') => screen.getByRole('radio', { name })

/** Press the form's save and return the configuration fragment it sent. */
function save(onSave: ReturnType<typeof renderForm>) {
  fireEvent.click(screen.getByRole('button', { name: 'save' }))
  return onSave.mock.calls[onSave.mock.calls.length - 1][0]
}

const API_AGENT = { runtime: 'api' }
const CLI_AGENT = { runtime: 'cli', provider: 'claude' }

describe('Runtime profile: what it shows', () => {
  it('an agent with no profile shows Full, and an untouched agent saves no profile', async () => {
    const mod = await load('local', apiMock({ configuration: API_AGENT }))
    const onSave = renderForm(mod, API_AGENT)
    await waitFor(() => expect(radio('Full')).toBeChecked())
    expect(radio('Lean')).not.toBeChecked()
    expect(radio('Lean')).toBeEnabled()
    expect(save(onSave)).toEqual({ runtime: 'api' })
  })

  it('an agent saved as full shows Full', async () => {
    const configuration = { ...API_AGENT, runtime_profile: 'full' }
    const mod = await load('local', apiMock({ configuration }))
    renderForm(mod, configuration)
    await waitFor(() => expect(radio('Full')).toBeChecked())
    expect(radio('Lean')).not.toBeChecked()
  })

  it('an agent saved as lean shows Lean', async () => {
    const configuration = { ...API_AGENT, runtime_profile: 'lean' }
    const mod = await load('local', apiMock({ configuration }))
    renderForm(mod, configuration)
    await waitFor(() => expect(radio('Lean')).toBeChecked())
    expect(radio('Full')).not.toBeChecked()
  })

  it('labels it "Runtime profile", apart from the Runtime choice, with the help text under it', async () => {
    const mod = await load('local', apiMock({ configuration: API_AGENT }))
    renderForm(mod, API_AGENT)
    expect(await screen.findByRole('radiogroup', { name: 'Runtime profile' })).toBeInTheDocument()
    expect(screen.getByLabelText('Runtime')).toBeInTheDocument()
    expect(
      screen.getByText(
        "Lean agents answer inside a product's app through its server key: only the tools you grant, no Auto personality.",
      ),
    ).toBeInTheDocument()
  })

  it('shows in the hosted edition too, where the Runtime group does not render', async () => {
    const mod = await load('saas', apiMock({ configuration: API_AGENT }))
    renderForm(mod, API_AGENT)
    await waitFor(() => expect(radio('Full')).toBeChecked())
    expect(screen.queryByTestId('runtime-section')).not.toBeInTheDocument()
  })
})

describe('Runtime profile: saving through the existing save', () => {
  it('choosing Lean adds runtime_profile "lean" to what an api agent already sends, and nothing else', async () => {
    const mod = await load('saas', apiMock({ configuration: API_AGENT }))
    const onSave = renderForm(mod, API_AGENT)
    await waitFor(() => expect(radio('Lean')).toBeEnabled())
    fireEvent.click(radio('Lean'))
    expect(radio('Lean')).toBeChecked()
    expect(save(onSave)).toEqual({ ...fragmentBefore(mod, API_AGENT), runtime_profile: 'lean' })
  })

  it('choosing Lean adds runtime_profile "lean" to what a cli agent already sends, and nothing else', async () => {
    const mod = await load('local', apiMock({ configuration: CLI_AGENT }))
    const onSave = renderForm(mod, CLI_AGENT)
    await waitFor(() => expect(radio('Lean')).toBeEnabled())
    fireEvent.click(radio('Lean'))
    expect(save(onSave)).toEqual({ ...fragmentBefore(mod, CLI_AGENT), runtime_profile: 'lean' })
  })

  it('choosing Full on a lean agent saves runtime_profile "full"', async () => {
    const configuration = { ...API_AGENT, runtime_profile: 'lean' }
    const mod = await load('local', apiMock({ configuration }))
    const onSave = renderForm(mod, configuration)
    await waitFor(() => expect(radio('Lean')).toBeChecked())
    fireEvent.click(radio('Full'))
    expect(radio('Full')).toBeChecked()
    expect(save(onSave)).toEqual({ ...fragmentBefore(mod, configuration), runtime_profile: 'full' })
  })
})

describe('Runtime profile: a system agent', () => {
  it('shows Full, disabled, with the reason, whatever its configuration says, and cannot save Lean', async () => {
    const configuration = { ...API_AGENT, runtime_profile: 'lean' }
    const mod = await load('local', apiMock({ is_system_agent: true, configuration }))
    const onSave = renderForm(mod, configuration)
    // the choice is disabled while the agent loads too: wait for the system agent's reason first
    expect(await screen.findByText('A system agent always runs Full.')).toBeInTheDocument()
    expect(radio('Full')).toBeChecked()
    expect(radio('Full')).toBeDisabled()
    expect(radio('Lean')).toBeDisabled()
    fireEvent.click(radio('Lean'))
    expect(radio('Lean')).not.toBeChecked()
    expect(save(onSave)).not.toHaveProperty('runtime_profile')
  })
})

describe('Runtime profile: a failed read', () => {
  it('keeps the choice disabled, shows nothing as chosen, and says why', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    const mod = await load('saas', apiMock(new Error('agent read failed')))
    const onSave = renderForm(mod, API_AGENT)
    expect(await screen.findByTestId('runtime-profile-locked')).toHaveTextContent("Couldn't load this agent's runtime profile")
    expect(radio('Full')).toBeDisabled()
    expect(radio('Full')).not.toBeChecked()
    expect(radio('Lean')).not.toBeChecked()
    fireEvent.click(radio('Lean'))
    expect(save(onSave)).not.toHaveProperty('runtime_profile')
  })
})
