/**
 * Issue #830 — a key that failed save-time validation is stored
 * `is_active=false` and never resolves, but the "Your API Keys" list used
 * to render it exactly like a healthy one: a masked key and no indication
 * anything was wrong. This row must mark it Inactive and keep the existing
 * Test action (`POST /api/keys/{id}/test`) reachable.
 */
import { describe, it, expect, vi } from 'vitest'
import { fireEvent, render, screen } from '@testing-library/react'

import { ApiKeyProviderRow } from '../ApiKeyProviderRow'
import type { ApiKeyOut } from '../api-keys-types'

const PROVIDER = { value: 'openai', label: 'OpenAI' }

function key(overrides: Partial<ApiKeyOut> = {}): ApiKeyOut {
  return {
    id: 1,
    provider: 'openai',
    display_name: null,
    masked_key: 'sk-ab...cd12',
    is_active: true,
    last_used_at: null,
    usage_count: 0,
    created_at: '2026-10-05T00:00:00Z',
    validation: null,
    ...overrides,
  }
}

function renderRow(keys: ApiKeyOut[], onTest = vi.fn()) {
  render(
    <ApiKeyProviderRow
      provider={PROVIDER}
      keys={keys}
      byokOn={false}
      hasPlatformKey={false}
      testingKeyId={null}
      onToggleByok={vi.fn()}
      onTest={onTest}
      onDelete={vi.fn()}
    />,
  )
  return { onTest }
}

describe('ApiKeyProviderRow (issue #830)', () => {
  it('marks a key that failed validation as Inactive and keeps Test reachable', () => {
    const inactiveKey = key({ id: 7, is_active: false })
    const { onTest } = renderRow([inactiveKey])

    expect(screen.getByTestId('key-inactive-openai')).toHaveTextContent('Inactive')
    expect(screen.getByText('No key available')).toBeInTheDocument() // never trusted — no active source

    const testButton = screen.getByTitle('Test key')
    expect(testButton).not.toBeDisabled()
    fireEvent.click(testButton)
    expect(onTest).toHaveBeenCalledWith(7)
  })

  it('shows no Inactive badge for a healthy, active key', () => {
    renderRow([key({ id: 9, is_active: true })])

    expect(screen.queryByTestId('key-inactive-openai')).not.toBeInTheDocument()
    expect(screen.getByTitle('Test key')).not.toBeDisabled()
  })

  it('disables Test only while that specific key is being tested', () => {
    render(
      <ApiKeyProviderRow
        provider={PROVIDER}
        keys={[key({ id: 3 })]}
        byokOn={false}
        hasPlatformKey={false}
        testingKeyId={3}
        onToggleByok={vi.fn()}
        onTest={vi.fn()}
        onDelete={vi.fn()}
      />,
    )

    expect(screen.getByTitle('Test key')).toBeDisabled()
  })
})
