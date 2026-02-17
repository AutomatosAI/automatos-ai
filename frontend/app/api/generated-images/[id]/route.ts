import { NextRequest } from 'next/server'

const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params

  const backendResponse = await fetch(
    `${BACKEND_URL}/api/generated-images/${id}`,
    { headers: { 'Accept': 'image/*' } }
  )

  if (!backendResponse.ok) {
    return new Response('Image not found', { status: 404 })
  }

  const body = backendResponse.body
  const contentType = backendResponse.headers.get('Content-Type') || 'image/png'

  return new Response(body, {
    status: 200,
    headers: {
      'Content-Type': contentType,
      'Cache-Control': 'public, max-age=86400, immutable',
    },
  })
}
