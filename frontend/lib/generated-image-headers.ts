// F179: a public generated image never renders as a page on the app's origin.
// The browser keeps the stored type (nosniff) and runs nothing (a sandboxed CSP
// that loads nothing); a raster image shows inline and anything else downloads.
// The same headers as orchestrator/api/generated_images.py.

const RASTER_IMAGE_TYPES = new Set(['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/webp'])
// The canonical uuid the image store mints; nothing else is forwarded to the backend.
const IMAGE_ID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/

export const IMAGE_CACHE_CONTROL = 'public, max-age=86400, immutable'

export function isImageId(id: string): boolean {
  return IMAGE_ID.test(id)
}

export function imageResponseHeaders(contentType: string): Record<string, string> {
  const base = contentType.split(';')[0].trim().toLowerCase()
  return {
    'Content-Type': contentType,
    'Cache-Control': IMAGE_CACHE_CONTROL,
    'Content-Disposition': RASTER_IMAGE_TYPES.has(base) ? 'inline' : 'attachment',
    'X-Content-Type-Options': 'nosniff',
    'Content-Security-Policy': "default-src 'none'; sandbox",
  }
}
