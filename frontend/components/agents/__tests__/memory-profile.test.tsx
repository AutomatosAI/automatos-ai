/**
 * PRD-274 Wave 4 — the "Memory profile" choice of a saved agent, under its runtime profile on the
 * Model tab of its configuration modal.
 *
 * The picker lists the profiles `GET /api/v1/memory/profiles` answers, by name, with General when
 * the list does not hold it. An agent with no `configuration.memory_profile` follows its workspace
 * until a person picks one; the pick rides the form, so the modal's existing save sends
 * `memory_profile` only then, beside everything it already sent.
 *
 * The <Select> is mocked as a native <select> (Radix's listbox needs pointer capture and
 * scrollIntoView, which jsdom does not implement). The hosted edition is used so it is the only
 * select on the page: the local-only Runtime group has selects of its own.
 */
import { useState, type ReactNode } from 'react'
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react'

type WrapperProps = { children?: ReactNode }
type MockSelectProps = WrapperProps & { value: string; onValueChange: (value: string) => void; disabled?: boolean }

vi.mock('@/components/ui/select', () => ({
  Select: ({ value, onValueChange, disabled, children }: MockSelectProps) => (
    <select value={value} disabled={disabled} onChange={(e) => onValueChange(e.target.value)}>
      {children}
    </select>
  ),
  SelectTrigger: () => null,
  SelectValue: () => null,
  SelectContent: ({ children }: WrapperProps) => <>{children}</>,
  SelectItem: ({ value, children }: WrapperProps & { value: string }) => <option value={value}>{children}</option>,
}))

const AGENT_ID = 7
const PROFILES_PATH = '/api/v1/memory/profiles'

type Detail = Record<string, unknown>

function profile(slug: string, name: string, description: string): Detail {
  return { slug, name, description, focus: '', keep: [], skip: [], examples: [], seeded: true }
}

const CUSTOMER = profile('customer', 'Customer', 'For an agent that serves one customer at a time.')
const GENERAL = profile('general', 'General', 'What Auto keeps today.')
const INCIDENT = profile('incident', 'Incident', '')

/** apiClient.request by path: the saved agent, and the memory profiles (or their failure). */
function apiMock(saved: Detail, profiles: Detail | Error) {
  return vi.fn(async (path: string) => {
    if (path === `/api/agents/${AGENT_ID}`) return saved
    if (path === PROFILES_PATH) {
      if (profiles instanceof Error) throw profiles
      return profiles
    }
    throw new Error(`unexpected path ${path}`)
  })
}

async function load(request: ReturnType<typeof apiMock>) {
  vi.resetModules()
  vi.doMock('@/lib/auth-edition', () => ({ authEdition: 'saas', isLocal: false, isSaaS: true }))
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

/** Press the form's save and return the configuration fragment it sent. */
function save(onSave: ReturnType<typeof renderForm>) {
  fireEvent.click(screen.getByRole('button', { name: 'save' }))
  return onSave.mock.calls[onSave.mock.calls.length - 1][0]
}

const picker = () => screen.getByRole('combobox') as HTMLSelectElement
const optionNames = () => within(picker()).getAllByRole('option').map((option) => option.textContent)
const shownName = () => picker().selectedOptions[0]?.textContent

/** The picker answers once both the agent and the list have; its options appear then. */
async function ready(name: string) {
  await screen.findByRole('option', { name })
  await waitFor(() => expect(picker()).toBeEnabled())
}

const API_AGENT = { runtime: 'api' }
const WITH_CUSTOMER = { ...API_AGENT, memory_profile: 'customer' }

describe('Memory profile: what it lists and shows', () => {
  it('lists the profiles by name, with General when the list does not hold it', async () => {
    const mod = await load(apiMock({ configuration: API_AGENT }, { profiles: [CUSTOMER, INCIDENT], enabled: true }))
    renderForm(mod, API_AGENT)
    await ready('Customer')
    expect(optionNames()).toEqual(['Workspace default', 'General', 'Customer', 'Incident'])
  })

  it('offers General once when the list holds it', async () => {
    const mod = await load(apiMock({ configuration: API_AGENT }, { profiles: [CUSTOMER, GENERAL, INCIDENT], enabled: true }))
    renderForm(mod, API_AGENT)
    await ready('Customer')
    expect(optionNames()).toEqual(['Workspace default', 'Customer', 'General', 'Incident'])
  })

  it('an agent with no profile follows the workspace default, says so, and saves no profile', async () => {
    const mod = await load(apiMock({ configuration: API_AGENT }, { profiles: [CUSTOMER], enabled: true }))
    const onSave = renderForm(mod, API_AGENT)
    await ready('Customer')
    expect(shownName()).toBe('Workspace default')
    expect(screen.getByTestId('memory-profile-help')).toHaveTextContent(
      "This agent uses the workspace's memory profile, or General when the workspace has none.",
    )
    expect(save(onSave)).not.toHaveProperty('memory_profile')
  })

  it("shows the agent's current profile, with that profile's description as the help line", async () => {
    const mod = await load(apiMock({ configuration: WITH_CUSTOMER }, { profiles: [CUSTOMER, INCIDENT], enabled: true }))
    renderForm(mod, WITH_CUSTOMER)
    await ready('Customer')
    expect(shownName()).toBe('Customer')
    expect(screen.getByTestId('memory-profile-help')).toHaveTextContent('For an agent that serves one customer at a time.')
    // the way back to the workspace's default is offered here too, first
    expect(optionNames()).toEqual(['Workspace default', 'General', 'Customer', 'Incident'])
  })

  it('shows no help line for a profile whose description is empty', async () => {
    const configuration = { ...API_AGENT, memory_profile: 'incident' }
    const mod = await load(apiMock({ configuration }, { profiles: [CUSTOMER, INCIDENT], enabled: true }))
    renderForm(mod, configuration)
    await ready('Incident')
    expect(shownName()).toBe('Incident')
    expect(screen.queryByTestId('memory-profile-help')).not.toBeInTheDocument()
  })

  it('says plainly that profiles are off for this install, and keeps the picker', async () => {
    const mod = await load(apiMock({ configuration: API_AGENT }, { profiles: [CUSTOMER], enabled: false }))
    renderForm(mod, API_AGENT)
    await ready('Customer')
    expect(screen.getByTestId('memory-profiles-off')).toHaveTextContent(
      'Memory profiles are off for this install: the choice is saved, and it applies once they are switched on.',
    )
    expect(picker()).toBeEnabled()
  })
})

describe('Memory profile: saving through the existing save', () => {
  it('choosing a profile adds exactly memory_profile to what the save already sends', async () => {
    const mod = await load(apiMock({ configuration: API_AGENT }, { profiles: [CUSTOMER, INCIDENT], enabled: true }))
    const onSave = renderForm(mod, API_AGENT)
    await ready('Customer')
    fireEvent.change(picker(), { target: { value: 'customer' } })
    expect(shownName()).toBe('Customer')
    expect(screen.getByTestId('memory-profile-help')).toHaveTextContent('For an agent that serves one customer at a time.')
    const before = mod.runtimeConfiguration(mod.runtimeFieldsFromConfiguration(API_AGENT))
    expect(save(onSave)).toEqual({ ...before, memory_profile: 'customer' })
  })

  it('choosing the workspace default again, before saving, sends no profile', async () => {
    const mod = await load(apiMock({ configuration: API_AGENT }, { profiles: [CUSTOMER], enabled: true }))
    const onSave = renderForm(mod, API_AGENT)
    await ready('Customer')
    const workspaceDefault = screen.getByRole('option', { name: 'Workspace default' }).getAttribute('value')
    fireEvent.change(picker(), { target: { value: 'customer' } })
    fireEvent.change(picker(), { target: { value: workspaceDefault } })
    expect(shownName()).toBe('Workspace default')
    expect(save(onSave)).not.toHaveProperty('memory_profile')
  })

  it("a system agent's picker stays open while its runtime profile is fixed", async () => {
    const mod = await load(apiMock({ is_system_agent: true, configuration: API_AGENT }, { profiles: [CUSTOMER], enabled: true }))
    const onSave = renderForm(mod, API_AGENT)
    expect(await screen.findByText('A system agent always runs Full.')).toBeInTheDocument()
    await ready('Customer')
    expect(screen.getByRole('radio', { name: 'Lean' })).toBeDisabled()
    fireEvent.change(picker(), { target: { value: 'customer' } })
    expect(save(onSave)).toEqual({ runtime: 'api', memory_profile: 'customer' })
  })
})

describe('Memory profile: back to the workspace default', () => {
  const DEFAULT_SENTENCE = "This agent uses the workspace's memory profile, or General when the workspace has none."
  const workspaceDefaultValue = () => screen.getByRole('option', { name: 'Workspace default' }).getAttribute('value')

  it('an agent with a saved profile offers Workspace default, first', async () => {
    const mod = await load(apiMock({ configuration: WITH_CUSTOMER }, { profiles: [CUSTOMER, GENERAL], enabled: true }))
    renderForm(mod, WITH_CUSTOMER)
    await ready('Customer')
    expect(optionNames()[0]).toBe('Workspace default')
    expect(optionNames()).toEqual(['Workspace default', 'Customer', 'General'])
  })

  it('choosing it on an agent with a saved profile saves memory_profile null, and nothing else changes', async () => {
    const mod = await load(apiMock({ configuration: WITH_CUSTOMER }, { profiles: [CUSTOMER], enabled: true }))
    const onSave = renderForm(mod, WITH_CUSTOMER)
    await ready('Customer')
    fireEvent.change(picker(), { target: { value: workspaceDefaultValue() } })
    expect(shownName()).toBe('Workspace default')
    // the default's sentence, and no description line beside it
    expect(screen.getAllByTestId('memory-profile-help')).toHaveLength(1)
    expect(screen.getByTestId('memory-profile-help')).toHaveTextContent(DEFAULT_SENTENCE)
    expect(screen.queryByText('For an agent that serves one customer at a time.')).not.toBeInTheDocument()
    const sent = save(onSave)
    expect(sent).toHaveProperty('memory_profile', null)
    expect(sent).toEqual({ ...mod.runtimeConfiguration(mod.runtimeFieldsFromConfiguration(WITH_CUSTOMER)), memory_profile: null })
  })

  it('an agent with none that picks it again sends no key', async () => {
    const mod = await load(apiMock({ configuration: API_AGENT }, { profiles: [CUSTOMER], enabled: true }))
    const onSave = renderForm(mod, API_AGENT)
    await ready('Customer')
    fireEvent.change(picker(), { target: { value: workspaceDefaultValue() } })
    expect(shownName()).toBe('Workspace default')
    expect(save(onSave)).not.toHaveProperty('memory_profile')
  })

  it("a stored null reads as Workspace default, with the default's sentence and no description", async () => {
    const configuration = { ...API_AGENT, memory_profile: null }
    const mod = await load(apiMock({ configuration }, { profiles: [CUSTOMER], enabled: true }))
    const onSave = renderForm(mod, configuration)
    await ready('Customer')
    expect(shownName()).toBe('Workspace default')
    expect(screen.getAllByTestId('memory-profile-help')).toHaveLength(1)
    expect(screen.getByTestId('memory-profile-help')).toHaveTextContent(DEFAULT_SENTENCE)
    fireEvent.change(picker(), { target: { value: workspaceDefaultValue() } })
    expect(save(onSave)).not.toHaveProperty('memory_profile')
  })
})

describe('Memory profile: a failed load', () => {
  it('shows the error state with a retry, and the runtime profile still saves', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    const request = apiMock({ configuration: API_AGENT }, new Error('profiles read failed'))
    const mod = await load(request)
    const onSave = renderForm(mod, API_AGENT)
    expect(await screen.findByRole('alert')).toHaveTextContent("Memory profiles couldn't be loaded")
    expect(screen.queryByRole('combobox')).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Try again' }))
    await waitFor(() => expect(request.mock.calls.filter(([path]) => path === PROFILES_PATH)).toHaveLength(2))
    await waitFor(() => expect(screen.getByRole('radio', { name: 'Lean' })).toBeEnabled())
    fireEvent.click(screen.getByRole('radio', { name: 'Lean' }))
    expect(save(onSave)).toEqual({ runtime: 'api', runtime_profile: 'lean' })
  })
})
