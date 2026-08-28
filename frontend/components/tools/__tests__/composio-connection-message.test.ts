import { describe, it, expect, vi } from 'vitest'
import {
    handleComposioConnectedMessage,
    isComposioConnectSuccess,
} from '../composio-connection-message'

/**
 * PRD-222 P222W2A-RVW-4 — the OAuth callback page posts COMPOSIO_CONNECTED on
 * BOTH the success and the error/denied path. The Tools page handler must gate
 * on the payload so a failed connection is not announced as "Connected!".
 */

function makeDeps() {
    return {
        queryClient: {
            invalidateQueries: vi.fn().mockResolvedValue(undefined),
            refetchQueries: vi.fn().mockResolvedValue(undefined),
        },
        setConnectingTool: vi.fn(),
        toast: vi.fn(),
    }
}

describe('isComposioConnectSuccess', () => {
    it('connected:true is a success', () => {
        expect(isComposioConnectSuccess({ app: 'GMAIL', status: 'active', connected: true })).toBe(true)
    })

    it('connected:false is a failure', () => {
        expect(isComposioConnectSuccess({ app: 'GMAIL', status: 'failed', connected: false })).toBe(false)
    })

    it('status error/failed is a failure even without an explicit connected flag', () => {
        expect(isComposioConnectSuccess({ app: 'GMAIL', status: 'error' })).toBe(false)
        expect(isComposioConnectSuccess({ app: 'GMAIL', status: 'failed' })).toBe(false)
    })

    it('a legacy message (no connected, non-error status) defaults to success', () => {
        expect(isComposioConnectSuccess({ app: 'GMAIL', status: 'active' })).toBe(true)
    })
})

describe('handleComposioConnectedMessage', () => {
    it('does NOT show the "Connected!" toast when connected:false', async () => {
        const deps = makeDeps()
        await handleComposioConnectedMessage(
            { type: 'COMPOSIO_CONNECTED', app: 'GMAIL', status: 'failed', connected: false },
            deps,
        )

        expect(deps.toast.mock.calls.some(([msg]) => msg === 'Connected!')).toBe(false)
        expect(deps.toast).toHaveBeenCalledWith(
            'Connection failed',
            expect.objectContaining({ description: expect.stringContaining('GMAIL') }),
        )
        // Failure path leaves spinner + refetch to the popup.closed poll.
        expect(deps.setConnectingTool).not.toHaveBeenCalled()
        expect(deps.queryClient.refetchQueries).not.toHaveBeenCalled()
    })

    it('does NOT show the "Connected!" toast when status:error', async () => {
        const deps = makeDeps()
        await handleComposioConnectedMessage(
            { type: 'COMPOSIO_CONNECTED', app: 'SLACK', status: 'error' },
            deps,
        )

        expect(deps.toast.mock.calls.some(([msg]) => msg === 'Connected!')).toBe(false)
        expect(deps.toast).toHaveBeenCalledWith('Connection failed', expect.anything())
    })

    it('shows the "Connected!" toast and refetches on connected:true', async () => {
        const deps = makeDeps()
        await handleComposioConnectedMessage(
            { type: 'COMPOSIO_CONNECTED', app: 'GMAIL', status: 'active', connected: true },
            deps,
        )

        expect(deps.toast).toHaveBeenCalledWith('Connected!', {
            description: 'GMAIL is now connected and ready to use.',
        })
        expect(deps.setConnectingTool).toHaveBeenCalledWith(null)
        expect(deps.queryClient.invalidateQueries).toHaveBeenCalledWith({ queryKey: ['tools'] })
        expect(deps.queryClient.refetchQueries).toHaveBeenCalledWith({ queryKey: ['tools'] })
        expect(deps.queryClient.invalidateQueries).toHaveBeenCalledWith({ queryKey: ['composio', 'connections'] })
    })

    it('reads the app label from event.data.app (not app_name), falling back to "App"', async () => {
        const named = makeDeps()
        await handleComposioConnectedMessage(
            { type: 'COMPOSIO_CONNECTED', app: 'NOTION', status: 'active', connected: true },
            named,
        )
        expect(named.toast).toHaveBeenCalledWith('Connected!', {
            description: 'NOTION is now connected and ready to use.',
        })

        const anon = makeDeps()
        await handleComposioConnectedMessage(
            { type: 'COMPOSIO_CONNECTED', status: 'active', connected: true },
            anon,
        )
        expect(anon.toast).toHaveBeenCalledWith('Connected!', {
            description: 'App is now connected and ready to use.',
        })
    })
})
