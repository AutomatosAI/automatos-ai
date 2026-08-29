/**
 * PRD-222 US-019 — the inline Composio connect card.
 *
 * Reuses the EXISTING OAuth flow (`useInitiateConnection` → the same
 * /api/composio/connect/{app} route + hosted popup the Tools page uses), then
 * bridges the `/tools/callback` page's existing `COMPOSIO_CONNECTED` postMessage
 * back into the card — no new event bus. These tests mock the popup + the
 * connection hooks and drive the callback contract to prove the result reflects
 * into chat state (connected / couldn't-connect), and that no new endpoint is
 * invented (the initiation hook is the only network call).
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react'

const { initiateMock, refetchMock, useConnectedAppsMock } = vi.hoisted(() => ({
  initiateMock: vi.fn(),
  refetchMock: vi.fn(async () => ({ data: [] as any[] })),
  useConnectedAppsMock: vi.fn(),
}))

vi.mock('@/hooks/use-composio-api', () => ({
  useInitiateConnection: () => ({ mutateAsync: initiateMock }),
  useConnectedApps: () => useConnectedAppsMock(),
}))

import { ConnectAppCard } from '../connect-app-card'

function setConnections(rows: Array<{ app_name: string; status: string }>) {
  useConnectedAppsMock.mockReturnValue({ data: rows, refetch: refetchMock })
}

/** Dispatch the callback page's COMPOSIO_CONNECTED message (same-origin). */
function postComposioMessage(payload: Record<string, unknown>) {
  act(() => {
    window.dispatchEvent(
      new MessageEvent('message', {
        data: { type: 'COMPOSIO_CONNECTED', ...payload },
        origin: window.location.origin,
      }),
    )
  })
}

describe('ConnectAppCard (US-019)', () => {
  let openSpy: ReturnType<typeof vi.spyOn>

  beforeEach(() => {
    initiateMock.mockReset()
    refetchMock.mockReset()
    refetchMock.mockResolvedValue({ data: [] })
    useConnectedAppsMock.mockReset()
    setConnections([])
    // jsdom's window.open is a no-op; return a fake, still-open popup.
    openSpy = vi.spyOn(window, 'open').mockReturnValue({ closed: false } as unknown as Window)
  })
  afterEach(() => vi.restoreAllMocks())

  it('names the app Auto asked for and offers to connect it', () => {
    render(<ConnectAppCard appName="GMAIL" displayName="Gmail" />)
    expect(screen.getByTestId('connect-app-card')).toHaveAttribute('data-app', 'GMAIL')
    expect(screen.getByTestId('connect-app-button')).toHaveTextContent(/connect gmail/i)
  })

  it('opens the EXISTING OAuth route + popup on connect (no new endpoint)', async () => {
    initiateMock.mockResolvedValueOnce({ redirect_url: 'https://oauth.example/authorize', app_name: 'GMAIL' })
    render(<ConnectAppCard appName="GMAIL" displayName="Gmail" />)

    fireEvent.click(screen.getByTestId('connect-app-button'))

    await waitFor(() =>
      expect(initiateMock).toHaveBeenCalledWith({
        appName: 'GMAIL',
        callbackUrl: `${window.location.origin}/tools/callback?connected=GMAIL`,
      }),
    )
    expect(openSpy).toHaveBeenCalledWith(
      'https://oauth.example/authorize',
      expect.stringContaining('Connect Gmail'),
      expect.stringContaining('width='),
    )
    expect(screen.getByTestId('connect-app-card')).toHaveAttribute('data-state', 'connecting')
  })

  it('reflects a successful callback back into the card', async () => {
    initiateMock.mockResolvedValueOnce({ redirect_url: 'https://oauth.example/authorize', app_name: 'GMAIL' })
    render(<ConnectAppCard appName="GMAIL" displayName="Gmail" />)

    fireEvent.click(screen.getByTestId('connect-app-button'))
    await waitFor(() =>
      expect(screen.getByTestId('connect-app-card')).toHaveAttribute('data-state', 'connecting'),
    )

    // The /tools/callback page posts this on success.
    postComposioMessage({ app: 'GMAIL', status: 'active', connected: true })

    await waitFor(() =>
      expect(screen.getByTestId('connect-app-card')).toHaveAttribute('data-state', 'connected'),
    )
    expect(screen.getByTestId('connect-app-done')).toHaveTextContent(/gmail is connected/i)
    expect(refetchMock).toHaveBeenCalled()
  })

  it('reflects a failed callback as a retryable error', async () => {
    initiateMock.mockResolvedValueOnce({ redirect_url: 'https://oauth.example/authorize', app_name: 'GMAIL' })
    render(<ConnectAppCard appName="GMAIL" displayName="Gmail" />)

    fireEvent.click(screen.getByTestId('connect-app-button'))
    await waitFor(() =>
      expect(screen.getByTestId('connect-app-card')).toHaveAttribute('data-state', 'connecting'),
    )

    postComposioMessage({ app: 'GMAIL', status: 'failed', connected: false })

    await waitFor(() =>
      expect(screen.getByTestId('connect-app-card')).toHaveAttribute('data-state', 'failed'),
    )
    expect(screen.getByTestId('connect-app-error')).toBeInTheDocument()
    // still retryable
    expect(screen.getByTestId('connect-app-button')).toBeInTheDocument()
  })

  it('ignores a callback for a different app', async () => {
    render(<ConnectAppCard appName="GMAIL" displayName="Gmail" />)
    postComposioMessage({ app: 'SLACK', status: 'active', connected: true })
    // never entered connecting, never flipped to connected
    expect(screen.getByTestId('connect-app-card')).toHaveAttribute('data-state', 'idle')
  })

  it('NO_AUTH apps (empty redirect_url) connect without a popup', async () => {
    initiateMock.mockResolvedValueOnce({ redirect_url: '', app_name: 'COMPOSIO_SEARCH' })
    render(<ConnectAppCard appName="COMPOSIO_SEARCH" displayName="Search" />)

    fireEvent.click(screen.getByTestId('connect-app-button'))

    await waitFor(() =>
      expect(screen.getByTestId('connect-app-card')).toHaveAttribute('data-state', 'connected'),
    )
    expect(openSpy).not.toHaveBeenCalled()
  })

  it('renders as connected when the app is already active in the list', () => {
    setConnections([{ app_name: 'GMAIL', status: 'active' }])
    render(<ConnectAppCard appName="GMAIL" displayName="Gmail" />)
    expect(screen.getByTestId('connect-app-card')).toHaveAttribute('data-state', 'connected')
    expect(screen.queryByTestId('connect-app-button')).toBeNull()
  })

  it('fires onConnected exactly once', async () => {
    const onConnected = vi.fn()
    initiateMock.mockResolvedValueOnce({ redirect_url: 'https://oauth.example/authorize', app_name: 'GMAIL' })
    render(<ConnectAppCard appName="GMAIL" displayName="Gmail" onConnected={onConnected} />)

    fireEvent.click(screen.getByTestId('connect-app-button'))
    await waitFor(() =>
      expect(screen.getByTestId('connect-app-card')).toHaveAttribute('data-state', 'connecting'),
    )
    postComposioMessage({ app: 'GMAIL', status: 'active', connected: true })
    postComposioMessage({ app: 'GMAIL', status: 'active', connected: true })

    await waitFor(() => expect(onConnected).toHaveBeenCalledWith('GMAIL'))
    expect(onConnected).toHaveBeenCalledTimes(1)
  })
})
