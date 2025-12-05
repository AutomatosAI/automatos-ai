/**
 * Workflow Streaming Proxy
 * 
 * This route proxies streaming requests to the FastAPI backend.
 * Uses proper streaming to ensure real-time updates reach the UI immediately.
 */

import { NextRequest } from 'next/server'

export const runtime = 'edge' // Edge runtime for true streaming

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
    
    const backendUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
    const streamUrl = `${backendUrl}/api/workflows/executions/${executionId}/stream/aisdk`
    
    console.log(`🚀 Proxying stream request to: ${streamUrl}`)
    
    // Fetch from backend with streaming
    const backendResponse = await fetch(streamUrl, {
      method: 'GET',
      headers: {
        'Accept': 'text/plain',
        'Cache-Control': 'no-cache',
      },
    })
    
    if (!backendResponse.ok) {
      const errorText = await backendResponse.text()
      console.error('❌ Backend stream error:', backendResponse.status, errorText)
      return new Response(
        JSON.stringify({ error: `Backend error: ${backendResponse.status}` }), 
        { status: backendResponse.status, headers: { 'Content-Type': 'application/json' } }
      )
    }
    
    // Return streaming response with proper headers for AI SDK
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
    console.error('❌ Stream proxy error:', error)
    return new Response(
      JSON.stringify({ error: error.message || 'Stream proxy failed' }),
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
  
  const backendUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
  const streamUrl = `${backendUrl}/api/workflows/executions/${executionId}/stream/aisdk`
  
  console.log(`🚀 GET stream request to: ${streamUrl}`)
  
  try {
    const backendResponse = await fetch(streamUrl, {
      method: 'GET',
      headers: {
        'Accept': 'text/plain',
        'Cache-Control': 'no-cache',
      },
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
    console.error('❌ GET stream error:', error)
    return new Response(
      JSON.stringify({ error: error.message }),
      { status: 500, headers: { 'Content-Type': 'application/json' } }
    )
  }
}

