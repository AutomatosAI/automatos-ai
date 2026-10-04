'use client'

/**
 * PRD-251B US-B301 — the brand kit's basics as the Brand kit tab edits them (it replaces
 * the BrandKitDialog): the kit loaded from GET /api/documents/brand-kit with the D5 fields
 * filled in, the prefill suggestions, the logo and the logo mark, and Save
 * (PUT /api/documents/brand-kit). The routes are the kit's own, unchanged.
 */
import { useCallback, useEffect, useState } from 'react'
import { toast } from 'sonner'

import { templateBlocksApi } from '@/components/documents/blocks/api'
import { useUploadedFontFaces } from '@/components/documents/blocks/BrandKitFonts'
import { toneWordsProblem } from '@/components/documents/blocks/BrandKitSocial'
import { useBrandImage } from '@/components/documents/blocks/useBrandImage'
import type { BrandKit, BrandSuggestions } from '@/components/documents/blocks/types'

const SOURCE_LABEL: Record<string, string> = {
  business_profile: 'your business profile',
  workspace: 'your workspace name',
  user: 'your account',
}
const NO_FONTS: BrandKit['font_files'] = []

/**
 * The PRD-251 D5 fields, filled when the kit comes from a backend that predates them.
 * `social_handles` is left as the server sent it: absent while Socials is off for the
 * workspace (PRD-251B US-B106), when no handles editor shows and the save carries none.
 */
export function withD5Fields(kit: BrandKit): BrandKit {
  return {
    ...kit,
    heading_font: kit.heading_font ?? '',
    font_files: kit.font_files ?? [],
    logo_mark_url: kit.logo_mark_url ?? '',
    logo_mark_path: kit.logo_mark_path ?? '',
    voice: {
      tone: kit.voice?.tone ?? [],
      banned_phrases: kit.voice?.banned_phrases ?? [],
      sign_off: kit.voice?.sign_off ?? '',
    },
  }
}

/** A 422 from PUT /brand-kit names each refused field and why: the kit's own check
 * (`{message, errors}`) or the request's (a list). */
export function saveErrorMessage(e: any): string {
  try {
    const detail = JSON.parse(e?.message ?? '')
    const errors: Array<{ loc?: unknown[]; msg?: string }> = Array.isArray(detail)
      ? detail
      : Array.isArray(detail?.errors) ? detail.errors : []
    if (errors.length) {
      return errors
        .map((err) => `${(err.loc ?? []).filter((part) => part !== 'body').join('.')}: ${(err.msg ?? '').replace(/^Value error, /, '')}`)
        .join('; ')
    }
  } catch {
    // Not a validation detail: the message as it came.
  }
  return e?.message || 'Failed to save brand kit'
}

/** Empty fields filled from what the platform already knows; a toast names where from. */
export function withSuggestions(kit: BrandKit, suggestions: BrandSuggestions): BrandKit {
  return {
    ...kit,
    name: kit.name || suggestions.name?.value || '',
    tagline: kit.tagline || suggestions.tagline?.value || '',
    logo_url: kit.logo_url || (kit.logo_path ? '' : suggestions.logo_url?.value || ''),
    company: {
      ...kit.company,
      name: kit.company.name || suggestions.company_name?.value || '',
      website: kit.company.website || suggestions.website?.value || '',
      email: kit.company.email || suggestions.email?.value || '',
    },
  }
}

export function useBrandKitForm() {
  const [kit, setKit] = useState<BrandKit | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const [suggestions, setSuggestions] = useState<BrandSuggestions>({})
  // Each load remounts the fields that keep their own typed text (the voice lists).
  const [loads, setLoads] = useState(0)

  const patch = useCallback((p: Partial<BrandKit>) => setKit((k) => (k ? { ...k, ...p } : k)), [])
  const patchCompany = (p: Partial<BrandKit['company']>) => setKit((k) => (k ? { ...k, company: { ...k.company, ...p } } : k))
  const logo = useBrandImage('logo', patch)
  const mark = useBrandImage('mark', patch)
  useUploadedFontFaces(kit?.font_files ?? NO_FONTS)

  useEffect(() => {
    templateBlocksApi
      .getBrandKit()
      .then((loaded) => {
        const k = withD5Fields(loaded)
        setKit(k)
        setLoads((n) => n + 1)
        logo.refresh(!!k.logo_path)
        mark.refresh(!!k.logo_mark_path)
      })
      .catch((e: any) => setLoadError(e?.message || 'Failed to load brand kit'))
    templateBlocksApi.getBrandSuggestions().then((r) => setSuggestions(r.suggestions || {})).catch(() => setSuggestions({}))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const applySuggestions = () => {
    if (!kit) return
    setKit(withSuggestions(kit, suggestions))
    const sources = Array.from(new Set(Object.values(suggestions).map((s) => SOURCE_LABEL[s.source] || s.source)))
    toast.success(`Filled empty fields from ${sources.join(' and ')}`)
  }

  const voiceProblem = kit ? toneWordsProblem(kit.voice.tone) : null
  const save = async () => {
    if (!kit || voiceProblem) return
    setSaving(true)
    try {
      // The stored files (logo_path, logo_mark_path, font_files) are server-managed;
      // the update route ignores them (validate_brand_kit strips them).
      setKit(withD5Fields(await templateBlocksApi.updateBrandKit(kit)))
      toast.success('Brand kit saved')
    } catch (e: any) {
      toast.error(saveErrorMessage(e))
    } finally {
      setSaving(false)
    }
  }

  return {
    kit, loadError, saving, suggestions, loads, logo, mark, voiceProblem,
    hasSuggestions: Object.keys(suggestions).length > 0,
    patch, patchCompany, applySuggestions, save,
  }
}

export type BrandKitForm = ReturnType<typeof useBrandKitForm>
