'use client'

/**
 * The Brand kit page's stored images: the logo, the logo mark (PRD-251 D5) and the logo's
 * dark and mono variants (PRD-255 FR-9), each uploaded, replaced and removed the same way.
 */
import { useCallback } from 'react'

import { useBrandImage } from '@/components/documents/blocks/useBrandImage'
import type { BrandKit } from '@/components/documents/blocks/types'

export function useBrandImages(patch: (p: Partial<BrandKit>) => void) {
  const logo = useBrandImage('logo', patch)
  const mark = useBrandImage('mark', patch)
  const dark = useBrandImage('dark', patch)
  const mono = useBrandImage('mono', patch)
  const { refresh: refreshLogo } = logo
  const { refresh: refreshMark } = mark
  const { refresh: refreshDark } = dark
  const { refresh: refreshMono } = mono

  /** Show each image ``kit`` has stored (none: cleared). */
  const showStored = useCallback((kit: BrandKit) => {
    void refreshLogo(!!kit.logo_path)
    void refreshMark(!!kit.logo_mark_path)
    void refreshDark(!!kit.logo_dark_path)
    void refreshMono(!!kit.logo_mono_path)
  }, [refreshLogo, refreshMark, refreshDark, refreshMono])

  return { logo, mark, dark, mono, showStored }
}
