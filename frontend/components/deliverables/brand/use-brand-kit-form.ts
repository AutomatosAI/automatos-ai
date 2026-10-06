'use client'

/**
 * PRD-251B US-B301 — the brand kit's basics as the Brand kit tab edits them (it replaces
 * the BrandKitDialog): the kit loaded from GET /api/documents/brand-kit with the D5 fields
 * filled in, the prefill suggestions, the logo, the logo mark and the logo's variants, the
 * colour roles (PRD-255), and Save (PUT /api/documents/brand-kit). `boardVersion` counts the
 * changes the page has stored (a save, an image or a font file uploaded or removed, a
 * colour role reset): after each, the brand board (PRD-255 US-010) reads the kit's stamp again
 * (F372; changes made elsewhere reach it through the stamp alone).
 */
import { useCallback, useEffect, useState, type Dispatch, type SetStateAction } from 'react'
import { toast } from 'sonner'

import { templateBlocksApi } from '@/components/documents/blocks/api'
import { useUploadedFontFaces } from '@/components/documents/blocks/BrandKitFonts'
import { toneWordsFrom, toneWordsProblem } from '@/components/documents/blocks/BrandKitSocial'
import type { BrandKit, BrandPaletteRole, BrandSuggestions } from '@/components/documents/blocks/types'
import { roleErrorsFrom, saveErrorMessage, type RoleErrors } from './save-errors'
import { useBrandImages } from './use-brand-images'
import { useBrandPalette } from './use-brand-palette'

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
      tone: toneWordsFrom(kit.voice?.tone),
      banned_phrases: kit.voice?.banned_phrases ?? [],
      sign_off: kit.voice?.sign_off ?? '',
    },
  }
}

/**
 * PRD-255: the kit as Save sends it. GET answers every effective colour role, the derived
 * ones too; sending those back would store them as set, and they would stop following the
 * kit's colours. So only the roles marked `set` go, and `palette_source` (the server's
 * answer, not a field) stays behind.
 */
export function kitToSave(kit: BrandKit): Partial<BrandKit> {
  const { palette, palette_source: sources, ...rest } = kit
  if (!palette) return rest
  if (!sources) return { ...rest, palette }
  const set = Object.fromEntries(Object.entries(palette).filter(([role]) => sources[role as BrandPaletteRole] === 'set'))
  return { ...rest, palette: set }
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

/** Save: PUT the kit; a refusal names the fields, and each colour role it names shows under its swatch. */
function useKitSave(
  kit: BrandKit | null,
  blocked: boolean,
  setKit: Dispatch<SetStateAction<BrandKit | null>>,
  setRoleErrors: (errors: RoleErrors) => void,
  onSaved: () => void,
) {
  const [saving, setSaving] = useState(false)
  const save = async () => {
    if (!kit || blocked) return
    setSaving(true)
    try {
      // The stored files (logo_path and its variants, logo_mark_path, font_files) are
      // server-managed; the update route ignores them (validate_brand_kit strips them).
      setKit(withD5Fields(await templateBlocksApi.updateBrandKit(kitToSave(kit))))
      setRoleErrors({})
      onSaved()
      toast.success('Brand kit saved')
    } catch (e: any) {
      setRoleErrors(roleErrorsFrom(e))
      toast.error(saveErrorMessage(e))
    } finally {
      setSaving(false)
    }
  }
  return { saving, save }
}

export function useBrandKitForm() {
  const [kit, setKit] = useState<BrandKit | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [suggestions, setSuggestions] = useState<BrandSuggestions>({})
  // Each load remounts the fields that keep their own typed text (the voice lists).
  const [loads, setLoads] = useState(0)
  const [boardVersion, setBoardVersion] = useState(0)
  const redrawBoard = useCallback(() => setBoardVersion((n) => n + 1), [])

  const patch = useCallback((p: Partial<BrandKit>) => setKit((k) => (k ? { ...k, ...p } : k)), [])
  const patchCompany = (p: Partial<BrandKit['company']>) => setKit((k) => (k ? { ...k, company: { ...k.company, ...p } } : k))
  // An image or a font file is stored by the server as it is uploaded or removed: the board redraws.
  const patchStored = useCallback((p: Partial<BrandKit>) => { patch(p); redrawBoard() }, [patch, redrawBoard])
  const images = useBrandImages(patchStored)
  const palette = useBrandPalette(setKit, redrawBoard)
  useUploadedFontFaces(kit?.font_files ?? NO_FONTS)

  useEffect(() => {
    templateBlocksApi
      .getBrandKit()
      .then((loaded) => {
        const k = withD5Fields(loaded)
        setKit(k)
        setLoads((n) => n + 1)
        images.showStored(k)
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
  const { saving, save } = useKitSave(kit, !!voiceProblem, setKit, palette.setRoleErrors, redrawBoard)

  return {
    kit, loadError, saving, suggestions, loads, boardVersion, ...images, palette, voiceProblem,
    hasSuggestions: Object.keys(suggestions).length > 0,
    patch, patchStored, patchCompany, applySuggestions, save,
  }
}

export type BrandKitForm = ReturnType<typeof useBrandKitForm>
