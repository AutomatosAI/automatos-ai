import { NextRequest } from 'next/server'

export const runtime = 'edge'

const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

// Get API key helper
function getApiKey(request: NextRequest): string {
  return process.env.NEXT_PUBLIC_API_KEY || 
         request.headers.get('x-api-key') ||
         'test_api_key_for_backend_validation_2025' // Fallback for Railway
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const apiKey = getApiKey(request)
    
    // Log for debugging (remove in production)
    if (!process.env.NEXT_PUBLIC_API_KEY) {
      console.warn('[Chat Proxy] Using fallback API key - set NEXT_PUBLIC_API_KEY in Railway')
    }
    
    // Forward to Python backend
    const backendResponse = await fetch(`${BACKEND_URL}/api/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'text/plain',
        'x-api-key': apiKey, // Always send API key
      },
      body: JSON.stringify(body),
    })

    if (!backendResponse.ok) {
      const errorText = await backendResponse.text()
      console.error('Backend error:', backendResponse.status, errorText)
      return new Response(
        JSON.stringify({ error: `Backend error: ${backendResponse.status}` }),
        { status: backendResponse.status, headers: { 'Content-Type': 'application/json' } }
      )
    }

    // Stream the response through
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
    console.error('Chat proxy error:', error)
    return new Response(
      JSON.stringify({ error: error.message || 'Chat proxy failed' }),
      { status: 500, headers: { 'Content-Type': 'application/json' } }
    )
  }
}

// Handle PATCH requests (for chat updates, voting, etc.)
export async function PATCH(request: NextRequest) {
  try {
    const body = await request.json()
    const apiKey = getApiKey(request)
    
    // Get the path from the request URL to forward to correct backend endpoint
    const url = new URL(request.url)
    const path = url.pathname.replace('/api/chat', '') // Remove /api/chat prefix
    
    // Forward to Python backend with the same path
    const backendResponse = await fetch(`${BACKEND_URL}/api/chat${path}`, {
      method: 'PATCH',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': apiKey,
      },
      body: JSON.stringify(body),
    })

    if (!backendResponse.ok) {
      const errorText = await backendResponse.text()
      console.error('Backend PATCH error:', backendResponse.status, errorText)
      return new Response(
        JSON.stringify({ error: `Backend error: ${backendResponse.status}` }),
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
      JSON.stringify({ error: error.message || 'Chat PATCH proxy failed' }),
      { status: 500, headers: { 'Content-Type': 'application/json' } }
    )
  }
}