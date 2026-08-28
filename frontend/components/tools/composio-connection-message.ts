import type { QueryClient } from '@tanstack/react-query'

/**
 * Shape of the `COMPOSIO_CONNECTED` popup message posted by
 * `app/tools/callback/page.tsx` back to `window.opener`. The callback posts it
 * on BOTH the success and the error/denied path (the PRD-222 inline connect
 * card needs the failure signal), so the app label lives under `app` and the
 * outcome is carried by `connected` / `status`.
 */
export interface ComposioConnectedMessage {
    type?: string
    app?: string
    status?: string
    connectionId?: string
    connected?: boolean
}

export interface ComposioMessageDeps {
    queryClient: Pick<QueryClient, 'invalidateQueries' | 'refetchQueries'>
    setConnectingTool: (tool: string | null) => void
    toast: (message: string, opts?: { description?: string }) => void
}

/**
 * A COMPOSIO_CONNECTED message is only a success when the payload says so.
 * Treating any such message as success announces a false "Connected!" on a
 * denied/failed OAuth (the callback posts the same message type on failure).
 */
export function isComposioConnectSuccess(data: ComposioConnectedMessage): boolean {
    return data.connected !== false && data.status !== 'error' && data.status !== 'failed'
}

/**
 * React to a COMPOSIO_CONNECTED popup message on the Tools page. Success clears
 * the connecting spinner, refetches the tool/connection queries and confirms;
 * failure only corrects the message — the `popup.closed` poll in
 * `handleToolConnect` owns spinner + refetch cleanup on the failure path.
 */
export async function handleComposioConnectedMessage(
    data: ComposioConnectedMessage,
    { queryClient, setConnectingTool, toast }: ComposioMessageDeps,
): Promise<void> {
    const appLabel = data.app || 'App'

    if (!isComposioConnectSuccess(data)) {
        toast('Connection failed', { description: `${appLabel} didn't connect. Please try again.` })
        return
    }

    setConnectingTool(null)
    await queryClient.invalidateQueries({ queryKey: ['tools'] })
    await queryClient.refetchQueries({ queryKey: ['tools'] })
    await queryClient.invalidateQueries({ queryKey: ['composio', 'connections'] })
    toast('Connected!', { description: `${appLabel} is now connected and ready to use.` })
}
