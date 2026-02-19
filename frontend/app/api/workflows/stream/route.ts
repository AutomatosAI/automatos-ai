/**
 * Workflow Streaming Proxy
 *
 * This route proxies streaming requests to the FastAPI backend.
 * Uses proper streaming to ensure real-time updates reach the UI immediately.
 */

import { NextRequest } from 'next/server'

export const runtime = 'edge' // Edge runtime for true streaming

// SECURITY: UUID v4 validation to prevent SSRF via path injection (OWASP A10:2021 - SSRF)
const UUID_REGEX = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i

function isValidUUID(value: string): boolean {
  return UUID_REGEX.test(value)
}

function getAuthHeaders(request: NextRequest): Record<string, string> {
  const headers: Record<string, string> = {
    'Accept': 'text/plain',
    'Cache-Control': 'no-cache',
  }
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

    // SECURITY: Validate executionId is a UUID to prevent SSRF path injection
    if (!isValidUUID(executionId)) {
      return new Response(JSON.stringify({ error: 'Invalid executionId format' }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' }
      })
    }

    const backendUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
    const streamUrl = `${backendUrl}/api/workflows/executions/${executionId}/stream/aisdk`

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

  // SECURITY: Validate executionId is a UUID to prevent SSRF path injection
  if (!isValidUUID(executionId)) {
    return new Response(JSON.stringify({ error: 'Invalid executionId format' }), {
      status: 400,
      headers: { 'Content-Type': 'application/json' }
    })
  }

  const backendUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
  const streamUrl = `${backendUrl}/api/workflows/executions/${executionId}/stream/aisdk`

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
