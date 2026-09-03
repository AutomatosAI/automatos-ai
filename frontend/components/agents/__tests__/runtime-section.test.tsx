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
      runtimeConfiguration({ runtime: 'cli', cli_provider: 'claude', cli_model: '  ', cli_working_directory: '' }),
    ).toEqual({ runtime: 'cli', provider: 'claude', model: null, working_directory: null })
    expect(
      runtimeConfiguration({ runtime: 'cli', cli_provider: '', cli_model: ' fable ', cli_working_directory: ' /w/repo ' }),
    ).toEqual({ runtime: 'cli', provider: 'claude', model: 'fable', working_directory: '/w/repo' })
  })

  it('reads the fields back from an agent configuration, defaulting anything missing', async () => {
    const { runtimeFieldsFromConfiguration, DEFAULT_RUNTIME_FIELDS } = await load('local')
    expect(runtimeFieldsFromConfiguration(undefined)).toEqual(DEFAULT_RUNTIME_FIELDS)
    expect(runtimeFieldsFromConfiguration({ runtime: 'api', model: 'gpt-4o' })).toEqual({
      ...DEFAULT_RUNTIME_FIELDS,
      cli_model: 'gpt-4o',
    })
    expect(
      runtimeFieldsFromConfiguration({ runtime: 'cli', provider: 'claude', model: 'opus', working_directory: '/w' }),
    ).toEqual({ runtime: 'cli', cli_provider: 'claude', cli_model: 'opus', cli_working_directory: '/w' })
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
    expect(screen.queryByLabelText(/Working directory/)).not.toBeInTheDocument()
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
    fireEvent.change(screen.getByLabelText(/Working directory/), { target: { value: '/w/repo' } })
    expect(onChange).toHaveBeenCalledWith('cli_model', 'fable')
    expect(onChange).toHaveBeenCalledWith('cli_working_directory', '/w/repo')
  })
})
