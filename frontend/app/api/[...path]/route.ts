import { NextRequest, NextResponse } from 'next/server'
import apiClient from '@/lib/api-client'

// This route handler attempts to proxy to the real backend
// If backend is down, it returns mock data directly
export async function GET(
  request: NextRequest,
  { params }: { params: { path: string[] } }
) {
  const path = params.path.join('/')
  const endpoint = `/api/${path}${request.nextUrl.search}`
  const backendUrl = `http://localhost:8000${endpoint}`
  
  try {
    const response = await fetch(backendUrl, {
      headers: {
        'Content-Type': 'application/json',
      },
      signal: AbortSignal.timeout(2000)
    })
    
    if (response.ok) {
      const data = await response.json()
      return NextResponse.json(data)
    }
    
    // Backend returned error - use mock
    const mockData = apiClient.getMockDataForEndpoint(endpoint)
    return NextResponse.json(mockData)
    
  } catch (error) {
    // Backend is down - return mock data directly
    const mockData = apiClient.getMockDataForEndpoint(endpoint)
    return NextResponse.json(mockData)
  }
}

export async function POST(
  request: NextRequest,
  { params }: { params: { path: string[] } }
) {
  const path = params.path.join('/')
  const endpoint = `/api/${path}`
  const backendUrl = `http://localhost:8000${endpoint}`
  
  try {
    const body = await request.json()
    const response = await fetch(backendUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(2000)
    })
    
    if (response.ok) {
      const data = await response.json()
      return NextResponse.json(data)
    }
    
    // Return mock success
    return NextResponse.json({ 
      success: true, 
      message: `Mock POST success for ${endpoint}`,
      data: body 
    })
    
  } catch (error) {
    return NextResponse.json({ 
      success: true, 
      message: `Mock POST success for ${endpoint}` 
    })
  }
}

export async function PUT(
  request: NextRequest,
  { params }: { params: { path: string[] } }
) {
  const path = params.path.join('/')
  const endpoint = `/api/${path}`
  const backendUrl = `http://localhost:8000${endpoint}`
  
  try {
    const body = await request.json()
    const response = await fetch(backendUrl, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(2000)
    })
    
    if (response.ok) {
      const data = await response.json()
      return NextResponse.json(data)
    }
    return NextResponse.json({ 
      success: true, 
      message: `Mock PUT success for ${endpoint}`,
      data: body 
    })
    
  } catch (error) {
    return NextResponse.json({ 
      success: true, 
      message: `Mock PUT success for ${endpoint}` 
    })
  }
}

export async function DELETE(
  request: NextRequest,
  { params }: { params: { path: string[] } }
) {
  const path = params.path.join('/')
  const endpoint = `/api/${path}`
  const backendUrl = `http://localhost:8000${endpoint}`
  
  try {
    const response = await fetch(backendUrl, {
      method: 'DELETE',
      headers: {
        'Content-Type': 'application/json',
      },
      signal: AbortSignal.timeout(2000)
    })
    
    if (response.ok) {
      const data = await response.json()
      return NextResponse.json(data)
    }
    return NextResponse.json({ 
      success: true, 
      message: `Mock DELETE success for ${endpoint}` 
    })
    
  } catch (error) {
    return NextResponse.json({ 
      success: true, 
      message: `Mock DELETE success for ${endpoint}` 
    })
  }
}