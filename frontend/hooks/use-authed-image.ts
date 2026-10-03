'use client'

/**
 * An image a route serves behind auth (a style reference, a post's AI option): a plain
 * <img src> cannot send the headers in SaaS, so it is fetched with them and shown from an
 * object URL, revoked when the path changes or the component goes. null while loading or
 * when the route has no file.
 */
import { useEffect, useState } from 'react'

import { templateBlocksApi } from '@/components/documents/blocks/api'

export function useAuthedImage(path: string | null): string | null {
  const [url, setUrl] = useState<string | null>(null)
  useEffect(() => {
    setUrl(null)
    if (!path) return undefined
    let alive = true
    let made: string | null = null
    templateBlocksApi
      .fetchBrandFileObjectUrl(path)
      .then((objectUrl) => {
        if (alive) {
          made = objectUrl
          setUrl(objectUrl)
        } else if (objectUrl) {
          URL.revokeObjectURL(objectUrl)
        }
      })
      .catch(() => undefined)
    return () => {
      alive = false
      if (made) URL.revokeObjectURL(made)
    }
  }, [path])
  return url
}
