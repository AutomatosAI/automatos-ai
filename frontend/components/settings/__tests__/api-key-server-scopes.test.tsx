/**
 * The create-key form's scopes (Settings › Widget SDK › API keys).
 *
 * The server accepts five scopes for a product's own backend (orchestrator/api/api_keys.py
 * `VALID_PERMISSIONS`: tasks:read, tasks:write, memory:read, memory:write, agents:chat), and the
 * form listed none of them, so such a key could only be minted by calling the API. The form now
 * offers them for a server key only, each ticked on its own: Select All leaves them out, and a
 * public key is never sent one.
 *
 * The fields run on the dialog's real `useCreateKeyForm`, and what would be sent is the real
 * `buildCreatePayload`. No key is minted. The Radix `<Select>` is a native `<select>` (the
 * api-key-manager-step-up.test.tsx precedent), and `@clerk/nextjs` is a stub because the form's
 * module imports the step-up hook.
 */
import React from 'react'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  COMMON_PERMISSIONS,
  SERVER_KEY_PERMISSIONS,
  type Permission,
} from '@/components/settings/api-keys/api-key-types'
import { KeyPermissionsField, KeyTypeField } from '@/components/settings/api-keys/create-key-fields'
import { buildCreatePayload, createFormError } from '@/components/settings/api-keys/create-key-payload'
import { useCreateKeyForm } from '@/components/settings/api-keys/use-create-key'

vi.mock('@clerk/nextjs', () => ({ useReverification: vi.fn() }))
vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }))

type WrapperProps = { children?: React.ReactNode }
vi.mock('@/components/ui/select', () => ({
  Select: ({ value, onValueChange, children }: WrapperProps & { value: string; onValueChange: (v: string) => void }) => (
    <select value={value} onChange={(e) => onValueChange(e.target.value)}>
      {children}
    </select>
  ),
  SelectTrigger: ({ children }: WrapperProps) => <>{children}</>,
  SelectValue: () => null,
  SelectContent: ({ children }: WrapperProps) => <>{children}</>,
  SelectItem: ({ value, children }: WrapperProps & { value: string }) => <option value={value}>{children}</option>,
}))

// orchestrator/api/api_keys.py VALID_PERMISSIONS, the PRD-274 scopes and the board's read scope.
const SERVER_KEY_SCOPES: Permission[] = ['tasks:read', 'tasks:write', 'memory:read', 'memory:write', 'agents:chat']
const SERVER_KEY_LABELS = ['Tasks: Read', 'Tasks: Write', 'Memory: Read', 'Memory: Write', 'Agents: Chat']
const COMMON_SCOPES = COMMON_PERMISSIONS.map((option) => option.value)
const NO_PERMISSION = 'Select at least one permission'

/** The key type and permissions fields, with what Create Key would send and why it would not. */
function Form() {
  const form = useCreateKeyForm()
  const filled = { ...form, formName: 'CI', formDomains: 'example.com' }
  return (
    <>
      <KeyTypeField form={form} />
      <KeyPermissionsField form={form} />
      <output data-testid="sent">{JSON.stringify(buildCreatePayload(filled).permissions)}</output>
      <output data-testid="refusal">{createFormError(filled) ?? ''}</output>
    </>
  )
}

const sent = () => JSON.parse(screen.getByTestId('sent').textContent ?? '[]') as Permission[]
const refusal = () => screen.getByTestId('refusal').textContent
const tick = (label: string) => fireEvent.click(screen.getByLabelText(label))
const press = (name: string) => fireEvent.click(screen.getByRole('button', { name }))
const chooseKeyType = (keyType: 'public' | 'server') =>
  fireEvent.change(screen.getByRole('combobox'), { target: { value: keyType } })

afterEach(() => {
  cleanup()
})

describe('the create-key form: server-key scopes', () => {
  it('lists the five scopes the server accepts for a backend, apart from the common ones', () => {
    expect(SERVER_KEY_PERMISSIONS.map((option) => option.value)).toEqual(SERVER_KEY_SCOPES)
    expect(COMMON_SCOPES.filter((scope) => SERVER_KEY_SCOPES.includes(scope))).toEqual([])
  })

  it('offers each one for a server key, the default type, unticked', () => {
    render(<Form />)

    for (const label of SERVER_KEY_LABELS) {
      expect(screen.getByLabelText(label)).toHaveAttribute('aria-checked', 'false')
    }
    expect(sent()).toEqual([])
    expect(refusal()).toBe(NO_PERMISSION)
  })

  it('says under each one what a key could do with it', () => {
    render(<Form />)

    for (const option of SERVER_KEY_PERMISSIONS) {
      expect(option.hint).toBeTruthy()
      expect(screen.getByLabelText(option.label)).toHaveAccessibleDescription(option.hint as string)
    }
  })

  it('sends the one that was ticked, and nothing else', () => {
    render(<Form />)

    tick('Tasks: Write')

    expect(screen.getByLabelText('Tasks: Write')).toHaveAttribute('aria-checked', 'true')
    expect(sent()).toEqual(['tasks:write'])
    expect(refusal()).toBe('')
  })

  it('Select All leaves them out', () => {
    render(<Form />)

    press('Select All')

    expect(sent()).toEqual(COMMON_SCOPES)
    for (const label of SERVER_KEY_LABELS) {
      expect(screen.getByLabelText(label)).toHaveAttribute('aria-checked', 'false')
    }
  })

  it('Select All keeps one already ticked, and Clear All unticks it', () => {
    render(<Form />)

    tick('Tasks: Write')
    press('Select All')
    expect(sent()).toEqual(['tasks:write', ...COMMON_SCOPES])

    press('Clear All')
    expect(sent()).toEqual([])
    expect(screen.getByLabelText('Tasks: Write')).toHaveAttribute('aria-checked', 'false')
  })
})

describe('the create-key form: a public key', () => {
  it('is not offered the server-key scopes', () => {
    render(<Form />)

    chooseKeyType('public')

    for (const label of SERVER_KEY_LABELS) {
      expect(screen.queryByLabelText(label)).not.toBeInTheDocument()
    }
    expect(screen.getByLabelText('Chat')).toBeInTheDocument()
  })

  it('is never sent one that was ticked before the type changed', () => {
    render(<Form />)

    tick('Chat')
    tick('Tasks: Write')
    tick('Agents: Chat')
    chooseKeyType('public')

    expect(sent()).toEqual(['chat'])
  })

  it('with only server-key scopes ticked has no permission, and says so', () => {
    render(<Form />)

    tick('Memory: Read')
    expect(refusal()).toBe('')

    chooseKeyType('public')

    expect(sent()).toEqual([])
    expect(refusal()).toBe(NO_PERMISSION)
  })

  it('gets them back, still ticked, when the type returns to server', () => {
    render(<Form />)

    tick('Memory: Write')
    chooseKeyType('public')
    chooseKeyType('server')

    expect(screen.getByLabelText('Memory: Write')).toHaveAttribute('aria-checked', 'true')
    expect(sent()).toEqual(['memory:write'])
  })
})
