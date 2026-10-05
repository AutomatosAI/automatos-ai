/**
 * Issue #830 — `POST /api/keys` validates the key before trusting it: a bad
 * key still comes back 201, stored `is_active=false`, with the outcome in
 * the response's `validation` field (orchestrator/api/user_api_keys.py).
 * The dialog used to ignore that field and always toast
 * "API key added successfully". It must now read `validation`, show the
 * provider's own error on a failure, and keep the dialog open so the user
 * can see it and fix the key — never claim success for a key that is
 * never going to be used.
 *
 * The provider <Select> is mocked as a plain native <select>: Radix's
 * listbox needs pointer-capture/scrollIntoView jsdom does not implement,
 * and the fix under test is in the mutation's onSuccess, not the picker.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import React from 'react'

const postMock = vi.hoisted(() => vi.fn())
const toastError = vi.hoisted(() => vi.fn())
const toastSuccess = vi.hoisted(() => vi.fn())

vi.mock('@/lib/api-client', () => ({
  apiClient: {
    get: () => Promise.reject(new Error('not used in this test')),
    post: (path: string, body: unknown) => postMock(path, body),
  },
}))
vi.mock('sonner', () => ({ toast: { success: toastSuccess, error: toastError } }))
type MockSelectProps = {
  value: string
  onValueChange: (value: string) => void
  children: React.ReactNode
}
type MockWrapperProps = { children: React.ReactNode }
type MockSelectItemProps = { value: string; children: React.ReactNode }

vi.mock('@/components/ui/select', () => ({
  Select: ({ value, onValueChange, children }: MockSelectProps) => (
    <select
      data-testid="provider-select"
      value={value}
      onChange={(e) => onValueChange(e.target.value)}
    >
      <option value="">Select provider</option>
      {children}
    </select>
  ),
  SelectTrigger: ({ children }: MockWrapperProps) => <>{children}</>,
  SelectValue: () => null,
  SelectContent: ({ children }: MockWrapperProps) => <>{children}</>,
  SelectItem: ({ value, children }: MockSelectItemProps) => <option value={value}>{children}</option>,
}))

import { AddApiKeyDialog } from '../AddApiKeyDialog'

const PROVIDERS = [{ value: 'openai', label: 'OpenAI' }]

function savedKey(overrides: Record<string, unknown> = {}) {
  return {
    id: 1,
    provider: 'openai',
    display_name: null,
    masked_key: 'sk-ba...key1',
    is_active: false,
    last_used_at: null,
    usage_count: 0,
    created_at: '2026-10-05T00:00:00Z',
    validation: null,
    ...overrides,
  }
}

function renderDialog() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  })
  return render(
    <QueryClientProvider client={client}>
      <AddApiKeyDialog providers={PROVIDERS} />
    </QueryClientProvider>,
  )
}

function openAndFillForm() {
  fireEvent.click(screen.getByRole('button', { name: /add api key/i }))
  fireEvent.change(screen.getByTestId('provider-select'), { target: { value: 'openai' } })
  fireEvent.change(screen.getByLabelText('API Key'), { target: { value: 'sk-bad-key' } })
}

describe('AddApiKeyDialog (issue #830)', () => {
  beforeEach(() => {
    postMock.mockReset()
    toastError.mockReset()
    toastSuccess.mockReset()
  })

  it('shows the provider error and keeps the dialog + key open on failed validation', async () => {
    postMock.mockResolvedValue(
      savedKey({
        validation: {
          valid: false,
          message: 'Invalid key: Incorrect API key provided',
          tested_at: '2026-10-05T00:00:00Z',
        },
      }),
    )
    renderDialog()
    openAndFillForm()

    fireEvent.click(screen.getByRole('button', { name: /^add key$/i }))

    await waitFor(() =>
      expect(toastError).toHaveBeenCalledWith(
        'Key saved but failed validation: Invalid key: Incorrect API key provided',
      ),
    )
    expect(toastSuccess).not.toHaveBeenCalled()

    // Dialog stays open (resetForm/close only run on a passed validation).
    expect(screen.getByRole('heading', { name: 'Add API Key' })).toBeInTheDocument()
    expect(screen.getByLabelText('API Key')).toHaveValue('sk-bad-key')
  })

  it('toasts success (not the generic error path) when validation passes', async () => {
    postMock.mockResolvedValue(
      savedKey({
        is_active: true,
        validation: { valid: true, message: 'API key is valid', tested_at: '2026-10-05T00:00:00Z' },
      }),
    )
    renderDialog()
    openAndFillForm()

    fireEvent.click(screen.getByRole('button', { name: /^add key$/i }))

    await waitFor(() => expect(toastSuccess).toHaveBeenCalledWith('API key added successfully'))
    expect(toastError).not.toHaveBeenCalled()
  })

  it('requires a provider and key before calling the API', () => {
    renderDialog()
    fireEvent.click(screen.getByRole('button', { name: /add api key/i }))

    fireEvent.click(screen.getByRole('button', { name: /^add key$/i }))

    expect(toastError).toHaveBeenCalledWith('Provider and API key are required')
    expect(postMock).not.toHaveBeenCalled()
  })
})
