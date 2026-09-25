'use client'

// The brand kit's font files (PRD-251 D5, S1.3). A woff2 is uploaded with the face
// it provides (family, weight, style) and stored like the logo; the same face
// uploaded again replaces it. A social render inlines every file, so a template's
// var(--brand-heading-font) can name an uploaded family. The uploaded faces are
// also registered in the browser, so the dialog's preview is set in them.
import { useEffect, useRef, useState } from 'react'
import { Loader2, Trash2, Upload } from 'lucide-react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { FieldHelp } from '@/components/ui/help-tooltip'
import { BRAND_FONTS_PATH, templateBlocksApi } from './api'
import type { BrandFontFile } from './types'

// The server's limits (modules/documents/brand_kit.py, brand_fonts.py).
export const MAX_FONT_FILES = 6
const MAX_FONT_MB = 2
const FONT_FAMILY = /^[A-Za-z0-9][A-Za-z0-9 _-]{0,63}$/
export const FONT_WEIGHTS: Record<number, string> = {
  100: 'Thin',
  200: 'Extra light',
  300: 'Light',
  400: 'Regular',
  500: 'Medium',
  600: 'Semi bold',
  700: 'Bold',
  800: 'Extra bold',
  900: 'Black',
}
const SELECT_CLASS = 'h-9 rounded-md border border-input bg-background px-2 text-sm'

/** "Brand Sans Bold italic". */
export function faceName(font: Pick<BrandFontFile, 'family' | 'weight' | 'style'>): string {
  return `${font.family} ${FONT_WEIGHTS[font.weight] ?? font.weight}${font.style === 'italic' ? ' italic' : ''}`
}

/**
 * Register the uploaded faces with the browser for the preview. The files need
 * auth headers, so each is fetched to an object URL for the FontFace API; a face
 * the browser refuses leaves the preview in its fallback font.
 */
export function useUploadedFontFaces(fonts: BrandFontFile[]) {
  useEffect(() => {
    if (typeof FontFace === 'undefined' || typeof document === 'undefined' || !document.fonts) return
    let cancelled = false
    const added: FontFace[] = []
    const urls: string[] = []
    fonts.forEach(async (font) => {
      const url = await templateBlocksApi.fetchBrandFileObjectUrl(`${BRAND_FONTS_PATH}/${font.id}`).catch(() => null)
      if (!url) return
      if (cancelled) {
        URL.revokeObjectURL(url)
        return
      }
      urls.push(url)
      try {
        const face = await new FontFace(font.family, `url(${url})`, { weight: String(font.weight), style: font.style }).load()
        if (cancelled) return
        document.fonts.add(face)
        added.push(face)
      } catch (e) {
        console.warn(`Brand kit preview: the browser could not load ${faceName(font)}`, e)
      }
    })
    return () => {
      cancelled = true
      added.forEach((face) => document.fonts.delete(face))
      urls.forEach((url) => URL.revokeObjectURL(url))
    }
  }, [fonts])
}

interface BrandKitFontsProps {
  fonts: BrandFontFile[]
  onChange: (fonts: BrandFontFile[]) => void
}

export function BrandKitFonts({ fonts, onChange }: BrandKitFontsProps) {
  const [family, setFamily] = useState('')
  const [weight, setWeight] = useState(400)
  const [style, setStyle] = useState<BrandFontFile['style']>('normal')
  const [busy, setBusy] = useState(false)
  const fileInput = useRef<HTMLInputElement | null>(null)
  const familyOk = FONT_FAMILY.test(family.trim())
  const full = fonts.length >= MAX_FONT_FILES

  const upload = async (file: File | undefined) => {
    if (!file || !familyOk) return
    const face = { family: family.trim(), weight, style }
    setBusy(true)
    try {
      const saved = await templateBlocksApi.uploadFont(file, face)
      onChange(saved.font_files)
      toast.success(`${faceName(face)} uploaded — name "${face.family}" as a font to use it`)
    } catch (e: any) {
      toast.error(e?.message || 'Font upload failed')
    } finally {
      setBusy(false)
      if (fileInput.current) fileInput.current.value = ''
    }
  }

  const remove = async (font: BrandFontFile) => {
    setBusy(true)
    try {
      const saved = await templateBlocksApi.deleteFont(font.id)
      onChange(saved.font_files)
      toast.success(`${faceName(font)} removed`)
    } catch (e: any) {
      toast.error(e?.message || 'Could not remove the font')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="rounded-md border p-3" data-testid="brand-kit-fonts">
      <Label className="flex items-center text-xs">
        Font files <FieldHelp id="deliverables.brand_kit.fonts" />
      </Label>
      {fonts.length > 0 && (
        <ul className="mt-2 space-y-1">
          {fonts.map((font) => (
            <li key={font.id} className="flex items-center justify-between gap-2 text-sm">
              <span className="min-w-0 truncate">
                <span className="font-medium">{faceName(font)}</span>
                {font.file_name && <span className="text-xs text-muted-foreground"> · {font.file_name}</span>}
              </span>
              <Button
                type="button"
                size="sm"
                variant="ghost"
                className="h-7 gap-1 text-destructive"
                disabled={busy}
                onClick={() => remove(font)}
                aria-label={`Remove ${faceName(font)}`}
              >
                <Trash2 className="h-3.5 w-3.5" />
              </Button>
            </li>
          ))}
        </ul>
      )}
      {full ? (
        <p className="mt-2 text-xs text-muted-foreground">
          {MAX_FONT_FILES} font files at most: remove one to add another.
        </p>
      ) : (
        <div className="mt-2 space-y-2">
          <div className="grid grid-cols-[minmax(0,1fr)_auto_auto] gap-2">
            <Input
              aria-label="Font family name"
              value={family}
              onChange={(e) => setFamily(e.target.value)}
              placeholder="Family name, e.g. Brand Sans"
            />
            <select aria-label="Font weight" value={weight} onChange={(e) => setWeight(Number(e.target.value))} className={SELECT_CLASS}>
              {Object.entries(FONT_WEIGHTS).map(([value, name]) => (
                <option key={value} value={value}>
                  {value} {name}
                </option>
              ))}
            </select>
            <select
              aria-label="Font style"
              value={style}
              onChange={(e) => setStyle(e.target.value === 'italic' ? 'italic' : 'normal')}
              className={SELECT_CLASS}
            >
              <option value="normal">Normal</option>
              <option value="italic">Italic</option>
            </select>
          </div>
          <input
            ref={fileInput}
            type="file"
            accept=".woff2,font/woff2"
            className="hidden"
            aria-label="Font file (woff2)"
            onChange={(e) => upload(e.target.files?.[0])}
          />
          <Button
            type="button"
            size="sm"
            variant="outline"
            className="gap-1.5"
            disabled={busy || !familyOk}
            onClick={() => fileInput.current?.click()}
          >
            {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Upload className="h-3.5 w-3.5" />}
            Upload woff2 ({MAX_FONT_MB} MB max)
          </Button>
        </div>
      )}
    </div>
  )
}
