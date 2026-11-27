/**
 * Workflow Streaming Proxy
 *
 * This route proxies streaming requests to the FastAPI backend.
 * Uses proper streaming to ensure real-time updates reach the UI immediately.
 */

import { NextRequest } from 'next/server'

import { isEnterprise } from '@/lib/edition-name'
import { forwardedSession } from '@/lib/enterprise/server-session'
import { executionIdPath } from '@/lib/workflow-execution-id'

export const runtime = 'edge' // Edge runtime for true streaming


function getAuthHeaders(request: NextRequest): Record<string, string> {
  const headers: Record<string, string> = {
    'Accept': 'text/plain',
    'Cache-Control': 'no-cache',
  }
  // PRD-277 US-503: enterprise passes the browser's session cookie and workspace on (the backend
  // call is a GET), and no key or bearer, which would stop the API reading the cookie.
  if (isEnterprise) return { ...headers, ...forwardedSession(request, 'GET') }
  const authHeader = request.headers.get('authorization')
  const apiKey = request.headers.get('x-api-key')
  const workspaceId =
    request.headers.get('x-workspace-id') ||
    request.headers.get('x-workspace') ||
    request.headers.get('X-Workspace-ID') ||
    request.headers.get('X-Workspace')
  if (authHeader) headers['Authorization'] = authHeader
  if (apiKey) headers['x-api-key'] = apiKey
  if (workspaceId) headers['X-Workspace-ID'] = workspaceId
  return headers
}

/**
 * Where GET reaches the API. PRD-277 US-505: in enterprise the browser-facing address is relative
 * (none is built in), so GET uses the server-side BACKEND_INTERNAL_URL, as POST does, which the
 * enterprise server requires at start-up (lib/enterprise/start-up.ts). Every other edition: as before.
 */
const LOCAL_BACKEND_URL = 'http://localhost:8000'

function getBackendUrlForGet(): string {
  if (isEnterprise) return process.env.BACKEND_INTERNAL_URL || LOCAL_BACKEND_URL
  return process.env.NEXT_PUBLIC_API_URL || LOCAL_BACKEND_URL
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const executionId = body.executionId

    if (!executionId) {
      return new Response(JSON.stringify({ error: 'executionId required' }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' }
      })
    }

    // SECURITY: only a positive whole number reaches the backend URL's path
    const executionPath = executionIdPath(executionId)
    if (executionPath === null) {
      return new Response(JSON.stringify({ error: 'Invalid executionId format' }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' }
      })
    }

    const backendUrl = // Server-side: BACKEND_INTERNAL_URL (container DNS, local edition) beats the
// browser-facing NEXT_PUBLIC_API_URL — inside the frontend container 'localhost'
// is the frontend itself. Unset in SaaS ⇒ identical to before. (PRD-209)
process.env.BACKEND_INTERNAL_URL || process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
    const streamUrl = `${backendUrl}/api/workflows/executions/${executionPath}/stream/aisdk`

    const backendResponse = await fetch(streamUrl, {
      method: 'GET',
      headers: getAuthHeaders(request),
    })

    if (!backendResponse.ok) {
      const errorText = await backendResponse.text()
      console.error('Backend stream error:', backendResponse.status, errorText)
      return new Response(
        JSON.stringify({ error: `Backend error: ${backendResponse.status}` }),
        { status: backendResponse.status, headers: { 'Content-Type': 'application/json' } }
      )
    }

    return new Response(backendResponse.body, {
      status: 200,
      headers: {
        'Content-Type': 'text/plain; charset=utf-8',
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Connection': 'keep-alive',
        'X-Accel-Buffering': 'no',
        'x-vercel-ai-data-stream': 'v1',
      },
    })

  } catch (error: any) {
    console.error('Stream proxy error:', error)
    return new Response(
      JSON.stringify({ error: 'Stream proxy failed' }),
      { status: 500, headers: { 'Content-Type': 'application/json' } }
    )
  }
}

// Also support GET for EventSource connections
export async function GET(request: NextRequest) {
  const url = new URL(request.url)
  const executionId = url.searchParams.get('executionId')

  if (!executionId) {
    return new Response(JSON.stringify({ error: 'executionId required' }), {
      status: 400,
      headers: { 'Content-Type': 'application/json' }
    })
  }

  // SECURITY: only a positive whole number reaches the backend URL's path
  const executionPath = executionIdPath(executionId)
  if (executionPath === null) {
    return new Response(JSON.stringify({ error: 'Invalid executionId format' }), {
      status: 400,
      headers: { 'Content-Type': 'application/json' }
    })
  }

  const backendUrl = getBackendUrlForGet()
  const streamUrl = `${backendUrl}/api/workflows/executions/${executionPath}/stream/aisdk`

  try {
    const backendResponse = await fetch(streamUrl, {
      method: 'GET',
      headers: getAuthHeaders(request),
    })

    if (!backendResponse.ok) {
      return new Response(
        JSON.stringify({ error: `Backend error: ${backendResponse.status}` }),
        { status: backendResponse.status, headers: { 'Content-Type': 'application/json' } }
      )
    }

    return new Response(backendResponse.body, {
      status: 200,
      headers: {
        'Content-Type': 'text/plain; charset=utf-8',
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Connection': 'keep-alive',
        'X-Accel-Buffering': 'no',
        'x-vercel-ai-data-stream': 'v1',
      },
    })
  } catch (error: any) {
    console.error('GET stream error:', error)
    return new Response(
      JSON.stringify({ error: 'Stream proxy failed' }),
      { status: 500, headers: { 'Content-Type': 'application/json' } }
    )
  }
}
