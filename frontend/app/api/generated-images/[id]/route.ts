import type { NextRequest } from 'next/server'

import { imageResponseHeaders, isImageId } from '@/lib/generated-image-headers'

const BACKEND_URL = // Server-side: BACKEND_INTERNAL_URL (container DNS, local edition) beats the
// browser-facing NEXT_PUBLIC_API_URL — inside the frontend container 'localhost'
// is the frontend itself. Unset in SaaS ⇒ identical to before. (PRD-209)
process.env.BACKEND_INTERNAL_URL || process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params

  // F179: only an image id reaches the backend, never a path to another route.
  if (!isImageId(id)) {
    return new Response('Image not found', { status: 404 })
  }

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
    headers: imageResponseHeaders(contentType),
  })
}
