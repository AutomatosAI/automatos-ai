import { NextRequest } from 'next/server'

export const runtime = 'edge'

const BACKEND_URL = // Server-side: BACKEND_INTERNAL_URL (container DNS, local edition) beats the
// browser-facing NEXT_PUBLIC_API_URL — inside the frontend container 'localhost'
// is the frontend itself. Unset in SaaS ⇒ identical to before. (PRD-209)
process.env.BACKEND_INTERNAL_URL || process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

// SECURITY: Allowlist of safe path patterns for PATCH proxy (OWASP A10:2021 - SSRF)
// Only allows paths like /history/{uuid}, /vote/{uuid}, /{uuid}
// Blocks path traversal (..), encoded sequences (%2e), and arbitrary destinations
const SAFE_CHAT_PATH_REGEX = /^\/[a-z][\w-]{0,30}\/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i

function isValidChatPath(path: string): boolean {
  if (!path || path === '') return true // Empty path is fine (PATCH to /api/chat itself)
  if (path.includes('..') || path.includes('%') || path.includes('\\')) return false
  return SAFE_CHAT_PATH_REGEX.test(path)
}

// Get API key helper — uses server-side API_KEY (not NEXT_PUBLIC_ since this is an edge route)
function getApiKey(request: NextRequest): string | null {
  return process.env.API_KEY ||
         process.env.ORCHESTRATOR_API_KEY ||
         request.headers.get('x-api-key') ||
         null
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const apiKey = getApiKey(request)
    const authHeader = request.headers.get('authorization')
    const workspaceId =
      request.headers.get('x-workspace-id') ||
      request.headers.get('x-workspace') ||
      request.headers.get('X-Workspace-ID') ||
      request.headers.get('X-Workspace')

    if (!process.env.API_KEY && !authHeader) {
      console.warn('[Chat Proxy] No API_KEY env var and no Authorization header')
    }

    // Build auth headers — send both JWT and API key (backend checks both independently)
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      'Accept': 'text/plain',
    }
    if (authHeader) {
      headers['Authorization'] = authHeader
    }
    if (apiKey) {
      // Only when a key exists — an absent key must not become the literal
      // string 'undefined' (the backend would then reject it as a bad API key).
      if (apiKey) headers['x-api-key'] = apiKey
    }
    if (workspaceId) {
      headers['X-Workspace-ID'] = workspaceId
    }

    // Forward to Python backend
    const backendResponse = await fetch(`${BACKEND_URL}/api/chat`, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
    })

    if (!backendResponse.ok) {
      const errorText = await backendResponse.text()
      console.error('Backend error:', backendResponse.status, errorText)
      return new Response(
        JSON.stringify({ error: 'Chat proxy failed' }),
        { status: backendResponse.status, headers: { 'Content-Type': 'application/json' } }
      )
    }

    // Forward routing headers from backend for frontend consumption
    const responseHeaders: Record<string, string> = {
      'Content-Type': 'text/plain; charset=utf-8',
      'Cache-Control': 'no-cache, no-store, must-revalidate',
      'Connection': 'keep-alive',
      'X-Accel-Buffering': 'no',
      'x-vercel-ai-data-stream': 'v1',
    }
    const routingHeaderNames = ['x-routing-agent-id', 'x-routing-confidence', 'x-routing-type', 'x-routing-reasoning', 'x-routing-request-id']
    for (const name of routingHeaderNames) {
      const value = backendResponse.headers.get(name)
      if (value) responseHeaders[name] = value
    }

    // Stream the response through
    return new Response(backendResponse.body, {
      status: 200,
      headers: responseHeaders,
    })
  } catch (error: any) {
    console.error('Chat proxy error:', error)
    return new Response(
      JSON.stringify({ error: 'Chat proxy failed' }),
      { status: 500, headers: { 'Content-Type': 'application/json' } }
    )
  }
}

// Handle PATCH requests (for chat updates, voting, etc.)
export async function PATCH(request: NextRequest) {
  try {
    const body = await request.json()
    const apiKey = getApiKey(request)
    const authHeader = request.headers.get('authorization')
    const workspaceId =
      request.headers.get('x-workspace-id') ||
      request.headers.get('x-workspace') ||
      request.headers.get('X-Workspace-ID') ||
      request.headers.get('X-Workspace')

    // Get the path from the request URL to forward to correct backend endpoint
    const url = new URL(request.url)
    const path = url.pathname.replace('/api/chat', '') // Remove /api/chat prefix

    // SECURITY: Validate path to prevent SSRF / path traversal (OWASP A10:2021 - SSRF)
    if (!isValidChatPath(path)) {
      return new Response(
        JSON.stringify({ error: 'Invalid request path' }),
        { status: 400, headers: { 'Content-Type': 'application/json' } }
      )
    }

    // Build auth headers — send both JWT and API key (backend checks both independently)
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    }
    if (authHeader) {
      headers['Authorization'] = authHeader
    }
    if (apiKey) {
      // Only when a key exists — an absent key must not become the literal
      // string 'undefined' (the backend would then reject it as a bad API key).
      if (apiKey) headers['x-api-key'] = apiKey
    }
    if (workspaceId) {
      headers['X-Workspace-ID'] = workspaceId
    }

    // Forward to Python backend with the same path
    const backendResponse = await fetch(`${BACKEND_URL}/api/chat${path}`, {
      method: 'PATCH',
      headers,
      body: JSON.stringify(body),
    })

    if (!backendResponse.ok) {
      const errorText = await backendResponse.text()
      console.error('Backend PATCH error:', backendResponse.status, errorText)
      return new Response(
        JSON.stringify({ error: 'Chat proxy failed' }),
        { status: backendResponse.status, headers: { 'Content-Type': 'application/json' } }
      )
    }

    return new Response(backendResponse.body, {
      status: backendResponse.status,
      headers: {
        'Content-Type': 'application/json',
      },
    })
  } catch (error: any) {
    console.error('Chat PATCH proxy error:', error)
    return new Response(
      JSON.stringify({ error: 'Chat proxy failed' }),
      { status: 500, headers: { 'Content-Type': 'application/json' } }
    )
  }
}
