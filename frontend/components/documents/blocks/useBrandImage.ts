'use client'

// The stored logo (PRD-242 S3), logo mark (PRD-251 D5) and the logo's dark and mono
// variants (PRD-255 FR-9): upload, remove, and an object URL to show it (a stored file
// needs auth headers a plain <img src> cannot send in SaaS). One hook for all, so they
// behave the same.
import { useCallback, useEffect, useRef, useState } from 'react'
import { toast } from 'sonner'
import { BRAND_LOGO_DARK_PATH, BRAND_LOGO_MARK_PATH, BRAND_LOGO_MONO_PATH, BRAND_LOGO_PATH, templateBlocksApi } from './api'
import type { BrandKit } from './types'

export type BrandImageKind = 'logo' | 'mark' | 'dark' | 'mono'

interface BrandImageSpec {
  path: string
  upload: (file: File) => Promise<BrandKit>
  remove: () => Promise<BrandKit>
  // The kit fields each call changes on the server.
  afterUpload: (saved: BrandKit) => Partial<BrandKit>
  afterRemove: (saved: BrandKit) => Partial<BrandKit>
  uploaded: string
  uploadFailed: string
  removed: string
  removeFailed: string
}

const SPECS: Record<BrandImageKind, BrandImageSpec> = {
  logo: {
    path: BRAND_LOGO_PATH,
    upload: (file) => templateBlocksApi.uploadLogo(file),
    remove: () => templateBlocksApi.deleteLogo(),
    afterUpload: (saved) => ({ logo_path: saved.logo_path, logo_url: saved.logo_url }),
    afterRemove: (saved) => ({ logo_path: saved.logo_path }),
    uploaded: 'Logo uploaded — it will appear in every document with a logo block',
    uploadFailed: 'Logo upload failed',
    removed: 'Logo removed',
    removeFailed: 'Could not remove the logo',
  },
  mark: {
    path: BRAND_LOGO_MARK_PATH,
    upload: (file) => templateBlocksApi.uploadLogoMark(file),
    remove: () => templateBlocksApi.deleteLogoMark(),
    afterUpload: (saved) => ({ logo_mark_path: saved.logo_mark_path, logo_mark_url: saved.logo_mark_url }),
    afterRemove: (saved) => ({ logo_mark_path: saved.logo_mark_path }),
    uploaded: 'Logo mark uploaded — social templates show it beside the brand name',
    uploadFailed: 'Logo mark upload failed',
    removed: 'Logo mark removed',
    removeFailed: 'Could not remove the logo mark',
  },
  dark: {
    path: BRAND_LOGO_DARK_PATH,
    upload: (file) => templateBlocksApi.uploadLogoVariant(BRAND_LOGO_DARK_PATH, file),
    remove: () => templateBlocksApi.deleteLogoVariant(BRAND_LOGO_DARK_PATH),
    afterUpload: (saved) => ({ logo_dark_path: saved.logo_dark_path }),
    afterRemove: (saved) => ({ logo_dark_path: saved.logo_dark_path }),
    uploaded: 'Logo for dark backgrounds uploaded',
    uploadFailed: 'Logo for dark backgrounds upload failed',
    removed: 'Logo for dark backgrounds removed',
    removeFailed: 'Could not remove the logo for dark backgrounds',
  },
  mono: {
    path: BRAND_LOGO_MONO_PATH,
    upload: (file) => templateBlocksApi.uploadLogoVariant(BRAND_LOGO_MONO_PATH, file),
    remove: () => templateBlocksApi.deleteLogoVariant(BRAND_LOGO_MONO_PATH),
    afterUpload: (saved) => ({ logo_mono_path: saved.logo_mono_path }),
    afterRemove: (saved) => ({ logo_mono_path: saved.logo_mono_path }),
    uploaded: 'One-colour logo uploaded',
    uploadFailed: 'One-colour logo upload failed',
    removed: 'One-colour logo removed',
    removeFailed: 'Could not remove the one-colour logo',
  },
}

export interface BrandImage {
  objectUrl: string | null
  busy: boolean
  refresh: (stored: boolean) => Promise<void>
  upload: (file: File | undefined) => Promise<void>
  remove: () => Promise<void>
}

export function useBrandImage(kind: BrandImageKind, patch: (p: Partial<BrandKit>) => void): BrandImage {
  const spec = SPECS[kind]
  const [objectUrl, setObjectUrl] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const shown = useRef<string | null>(null)

  const show = useCallback((url: string | null) => {
    if (shown.current) URL.revokeObjectURL(shown.current)
    shown.current = url
    setObjectUrl(url)
  }, [])

  // The last object URL goes with the dialog.
  useEffect(
    () => () => {
      if (shown.current) URL.revokeObjectURL(shown.current)
    },
    [],
  )

  const refresh = useCallback(
    async (stored: boolean) => {
      if (!stored) {
        show(null)
        return
      }
      try {
        show(await templateBlocksApi.fetchBrandFileObjectUrl(spec.path))
      } catch {
        show(null)
      }
    },
    [show, spec.path],
  )

  const upload = async (file: File | undefined) => {
    if (!file) return
    setBusy(true)
    try {
      const saved = await spec.upload(file)
      patch(spec.afterUpload(saved))
      await refresh(true)
      toast.success(spec.uploaded)
    } catch (e: any) {
      toast.error(e?.message || spec.uploadFailed)
    } finally {
      setBusy(false)
    }
  }

  const remove = async () => {
    setBusy(true)
    try {
      const saved = await spec.remove()
      patch(spec.afterRemove(saved))
      await refresh(false)
      toast.success(spec.removed)
    } catch (e: any) {
      toast.error(e?.message || spec.removeFailed)
    } finally {
      setBusy(false)
    }
  }

  return { objectUrl, busy, refresh, upload, remove }
}
