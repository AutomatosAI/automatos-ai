/**
 * PRD-222 US-013 — the power-up card.
 *
 * Renders at the powerup stage (and embedded for US-014's exhausted banner),
 * takes a masked key, posts it to the existing BYOK endpoint, and renders the
 * live validation result in-flow — success (full catalog unlocked) or the
 * provider's own error. Dismiss leaves chat usable. The raw key is masked, never
 * logged, and never written to a client-side store.
 *
 * Pure/mocked — apiClient, the workspace provider, and next/link are stubbed.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'

// Obviously-fake key (gitleaks-safe) — never a real credential format.
const FAKE_KEY = 'sk-or-FAKEKEY-not-a-real-key-000000'

const { postMock, refreshMock, useWorkspaceMock } = vi.hoisted(() => ({
  postMock: vi.fn(),
  refreshMock: vi.fn(async () => {}),
  useWorkspaceMock: vi.fn(),
}))

vi.mock('@/lib/api-client', () => ({ apiClient: { post: postMock } }))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspace: () => useWorkspaceMock(),
}))
vi.mock('next/link', () => ({
  default: ({ href, children, ...rest }: Record<string, unknown>) => (
    <a href={href as string} {...(rest as Record<string, string>)}>
      {children as React.ReactNode}
    </a>
  ),
}))

import { PowerUpCard } from '../power-up-card'

function mockWorkspace(
  stage: string,
  trial: { granted_usd: number; spent_usd: number; state: string } | null = {
    granted_usd: 5,
    spent_usd: 1.6,
    state: 'active',
  },
) {
  useWorkspaceMock.mockReturnValue({
    workspace: { onboarding: { stage, trial } },
    refreshWorkspace: refreshMock,
  })
}

async function connectWith(key: string) {
  fireEvent.change(screen.getByTestId('power-up-key-input'), {
    target: { value: key },
  })
  fireEvent.click(screen.getByTestId('power-up-connect'))
}

describe('PowerUpCard (US-013)', () => {
  beforeEach(() => {
    postMock.mockReset()
    refreshMock.mockClear()
    useWorkspaceMock.mockReset()
    localStorage.clear()
    sessionStorage.clear()
  })

  it('renders at the powerup stage with the continuation framing + trial balance', () => {
    mockWorkspace('powerup')
    render(<PowerUpCard />)

    expect(screen.getByTestId('power-up-card')).toBeInTheDocument()
    expect(
      screen.getByText(/keep auto running on your business/i),
    ).toBeInTheDocument()
    expect(screen.getByText(/Recommended: OpenRouter/i)).toBeInTheDocument()
    expect(screen.getByTestId('power-up-howto')).toHaveAttribute(
      'href',
      'https://openrouter.ai/keys',
    )
    expect(screen.getByTestId('power-up-trial')).toHaveTextContent('$3.40 of $5.00')
    expect(screen.getByTestId('power-up-settings-link')).toHaveAttribute(
      'href',
      '/settings',
    )
  })

  it('does not render outside the powerup stage', () => {
    mockWorkspace('questions')
    render(<PowerUpCard />)
    expect(screen.queryByTestId('power-up-card')).toBeNull()
  })

  it('renders embedded regardless of stage (US-014 reuse)', () => {
    mockWorkspace('exhausted' as string)
    render(<PowerUpCard embedded />)
    expect(screen.getByTestId('power-up-card')).toBeInTheDocument()
  })

  it('masks the key input', () => {
    mockWorkspace('powerup')
    render(<PowerUpCard />)
    expect(screen.getByTestId('power-up-key-input')).toHaveAttribute(
      'type',
      'password',
    )
  })

  it('success path shows the validated result and the full-catalog line', async () => {
    mockWorkspace('powerup')
    postMock.mockResolvedValueOnce({
      validation: { valid: true, message: 'API key is valid' },
    })
    render(<PowerUpCard />)

    await connectWith(FAKE_KEY)

    await waitFor(() =>
      expect(screen.getByTestId('power-up-result')).toHaveTextContent(
        'API key is valid',
      ),
    )
    expect(screen.getByTestId('power-up-result')).toHaveTextContent(
      /full model catalog is unlocked/i,
    )
    // posted to the existing BYOK endpoint with the raw key in the body only
    expect(postMock).toHaveBeenCalledWith('/api/keys', {
      provider: 'openrouter',
      api_key: FAKE_KEY,
    })
    expect(refreshMock).toHaveBeenCalled()
  })

  it('failure path renders the provider error text in the card', async () => {
    mockWorkspace('powerup')
    postMock.mockResolvedValueOnce({
      validation: { valid: false, message: 'Invalid key: 401 Unauthorized' },
    })
    render(<PowerUpCard />)

    await connectWith(FAKE_KEY)

    await waitFor(() =>
      expect(screen.getByTestId('power-up-result')).toHaveTextContent(
        'Invalid key: 401 Unauthorized',
      ),
    )
    expect(screen.getByTestId('power-up-result')).not.toHaveTextContent(
      /unlocked/i,
    )
  })

  it('dismiss keeps chat usable (card unmounts)', () => {
    mockWorkspace('powerup')
    render(<PowerUpCard />)
    fireEvent.click(screen.getByTestId('power-up-dismiss'))
    expect(screen.queryByTestId('power-up-card')).toBeNull()
  })

  it('never logs the raw key and never writes it to a client-side store', async () => {
    mockWorkspace('powerup')
    postMock.mockResolvedValueOnce({
      validation: { valid: true, message: 'API key is valid' },
    })
    const spies = (['log', 'info', 'warn', 'error', 'debug'] as const).map((m) =>
      vi.spyOn(console, m).mockImplementation(() => {}),
    )
    render(<PowerUpCard />)

    await connectWith(FAKE_KEY)
    await waitFor(() => expect(screen.getByTestId('power-up-result')).toBeInTheDocument())

    // no console call anywhere echoed the key
    for (const spy of spies) {
      for (const call of spy.mock.calls) {
        expect(JSON.stringify(call)).not.toContain(FAKE_KEY)
      }
    }
    // no client-side store holds the key
    const stores = [localStorage, sessionStorage]
    for (const store of stores) {
      for (let i = 0; i < store.length; i++) {
        const k = store.key(i)!
        expect(store.getItem(k) ?? '').not.toContain(FAKE_KEY)
      }
    }
    expect(JSON.stringify(localStorage)).not.toContain(FAKE_KEY)
    expect(JSON.stringify(sessionStorage)).not.toContain(FAKE_KEY)

    spies.forEach((s) => s.mockRestore())
  })

  afterEach(() => vi.restoreAllMocks())
})
