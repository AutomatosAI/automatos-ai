'use client'

import { useEffect } from 'react'
import { useRouter } from 'next/navigation'

import { CHAT_HREF } from '@/lib/ticket-links'

/**
 * PRD-261 A-04d — the core's home page, where a theme may draw its own: `/` has no page of its
 * own, it forwards to /chat. app/page.tsx forwards on the server whenever no theme of the style
 * can register a `/` page (Classic and Studio); under an installed theme, whose pages are known
 * only once its code loads in the browser, the page renders ThemeSlot with this as the core's
 * version, so the forward happens here, once the theme
 * host has said the theme draws no home of its own (or failed, or is below its width).
 */
export function CoreHomeForward(): null {
  const router = useRouter()
  useEffect(() => {
    router.replace(CHAT_HREF)
  }, [router])
  return null
}
