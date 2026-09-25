'use client'

import React, { useEffect, useRef, useState } from 'react'
import { ImagePlus, Loader2, Sparkles, Trash2 } from 'lucide-react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { FieldHelp } from '@/components/ui/help-tooltip'
import { templateBlocksApi } from './api'
import { BrandKitFonts, useUploadedFontFaces } from './BrandKitFonts'
import { BrandKitSocial, toneWordsProblem } from './BrandKitSocial'
import { useBrandImage } from './useBrandImage'
import type { BrandKit, BrandSuggestions } from './types'

interface BrandKitDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  onSaved?: (kit: BrandKit) => void
}

const HEX = /^#([0-9a-fA-F]{6})$/
const SOURCE_LABEL: Record<string, string> = {
  business_profile: 'your business profile',
  workspace: 'your workspace name',
  user: 'your account',
}
const NO_FONTS: BrandKit['font_files'] = []

// The PRD-251 D5 fields, filled when the kit comes from a backend that predates them
// (the frontend can deploy before the API).
function withD5Fields(kit: BrandKit): BrandKit {
  return {
    ...kit,
    heading_font: kit.heading_font ?? '',
    font_files: kit.font_files ?? [],
    logo_mark_url: kit.logo_mark_url ?? '',
    logo_mark_path: kit.logo_mark_path ?? '',
    social_handles: kit.social_handles ?? {},
    voice: { tone: kit.voice?.tone ?? [], banned_phrases: kit.voice?.banned_phrases ?? [] },
  }
}

function ColorField({ label, value, onChange }: { label: string; value: string; onChange: (v: string) => void }) {
  return (
    <div>
      <Label className="text-xs">{label}</Label>
      <div className="flex items-center gap-2">
        <input
          type="color"
          value={HEX.test(value) ? value : '#1a1a2e'}
          onChange={(e) => onChange(e.target.value)}
          className="h-9 w-10 cursor-pointer rounded border bg-transparent"
          aria-label={label}
        />
        <Input value={value} onChange={(e) => onChange(e.target.value)} className="font-mono text-sm" />
      </div>
    </div>
  )
}

// A 422 from PUT /brand-kit names each refused field and why ("voice.tone: give 3 to 5
// tone words"): the kit's own check ({message, errors}) or the request's (a list).
function saveErrorMessage(e: any): string {
  try {
    const detail = JSON.parse(e?.message ?? '')
    const errors: Array<{ loc?: unknown[]; msg?: string }> = Array.isArray(detail)
      ? detail
      : Array.isArray(detail?.errors)
        ? detail.errors
        : []
    if (errors.length) {
      return errors
        .map((err) => {
          const field = (err.loc ?? []).filter((part) => part !== 'body').join('.')
          return `${field}: ${(err.msg ?? '').replace(/^Value error, /, '')}`
        })
        .join('; ')
    }
  } catch {
    // Not a validation detail: the message as it came.
  }
  return e?.message || 'Failed to save brand kit'
}

// A live swatch of what the renderers will do with the palette (heading, rule, table head).
function KitPreview({ kit, logoUrl, markUrl }: { kit: BrandKit; logoUrl: string | null; markUrl: string | null }) {
  const mark = markUrl || kit.logo_mark_url || null
  return (
    <div className="rounded-md border bg-white p-3 text-[#1a1a2e]" style={{ fontFamily: kit.font_family || undefined, color: kit.text_color || undefined }}>
      {logoUrl ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={logoUrl} alt="Logo" className="mb-2 h-8 w-auto object-contain" />
      ) : kit.logo_url ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={kit.logo_url} alt="Logo" className="mb-2 h-8 w-auto object-contain" />
      ) : null}
      <div
        className="flex items-center gap-2 text-base font-bold"
        style={{ color: kit.primary_color, borderBottom: `2px solid ${kit.accent_color}`, fontFamily: kit.heading_font || undefined }}
      >
        {mark && (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={mark} alt="Logo mark" className="h-5 w-5 object-contain" />
        )}
        {kit.name || 'Your brand'}
      </div>
      <div className="mt-1 text-[11px]">{kit.tagline || 'Tagline'} · {kit.company.website || 'website'}</div>
      <div className="mt-2 grid grid-cols-2 text-[10px]">
        <div className="px-2 py-1 text-white" style={{ background: kit.primary_color }}>Table header</div>
        <div className="border px-2 py-1" style={{ borderColor: `${kit.secondary_color}55` }}>Cell</div>
      </div>
    </div>
  )
}

// Edit the workspace brand kit (PRD-167 S4 → PRD-242 S3 → PRD-251 D5): logo and logo
// mark uploads, prefill from what the platform already knows, every colour the
// renderers use, the body and heading fonts with uploaded font files, the social
// handles and the brand voice, and a live swatch. Opened from Template Studio and
// from the Socials tab.
export function BrandKitDialog({ open, onOpenChange, onSaved }: BrandKitDialogProps) {
  const [kit, setKit] = useState<BrandKit | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const [suggestions, setSuggestions] = useState<BrandSuggestions>({})
  // Each load remounts the fields that keep their own typed text (the voice lists).
  const [loads, setLoads] = useState(0)
  const logoInput = useRef<HTMLInputElement | null>(null)
  const markInput = useRef<HTMLInputElement | null>(null)

  const patch = (p: Partial<BrandKit>) => setKit((k) => (k ? { ...k, ...p } : k))
  const patchCompany = (p: Partial<BrandKit['company']>) =>
    setKit((k) => (k ? { ...k, company: { ...k.company, ...p } } : k))

  const logo = useBrandImage('logo', patch)
  const mark = useBrandImage('mark', patch)
  useUploadedFontFaces(kit?.font_files ?? NO_FONTS)

  useEffect(() => {
    if (!open) return
    setLoadError(null)
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
    templateBlocksApi
      .getBrandSuggestions()
      .then((r) => setSuggestions(r.suggestions || {}))
      .catch(() => setSuggestions({}))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open])

  const applySuggestions = () => {
    if (!kit) return
    const next: BrandKit = {
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
    setKit(next)
    const sources = Array.from(new Set(Object.values(suggestions).map((s) => SOURCE_LABEL[s.source] || s.source)))
    toast.success(`Filled empty fields from ${sources.join(' and ')}`)
  }

  const pickFile = (upload: (file: File | undefined) => Promise<void>) => (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    e.target.value = ''
    upload(file)
  }

  const voiceProblem = kit ? toneWordsProblem(kit.voice.tone) : null

  const save = async () => {
    if (!kit || voiceProblem) return
    setSaving(true)
    try {
      // The stored files (logo_path, logo_mark_path, font_files) are server-managed;
      // the update route ignores them (validate_brand_kit strips them).
      const saved = await templateBlocksApi.updateBrandKit(kit)
      toast.success('Brand kit saved')
      onSaved?.(saved)
      onOpenChange(false)
    } catch (e: any) {
      toast.error(saveErrorMessage(e))
    } finally {
      setSaving(false)
    }
  }

  const hasSuggestions = Object.keys(suggestions).length > 0

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[90vh] max-w-2xl overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center">
            Brand Kit <FieldHelp id="deliverables.brand_kit.title" />
          </DialogTitle>
          <DialogDescription>
            Applied to every document and social post rendered from a template — logo, colours, fonts, and the {'{{brand.*}}'} / {'{{company.*}}'} chips.
          </DialogDescription>
        </DialogHeader>

        {loadError ? (
          <p className="py-6 text-center text-sm text-destructive">{loadError}</p>
        ) : !kit ? (
          <div className="py-8 text-center text-sm text-muted-foreground">Loading…</div>
        ) : (
          <div className="grid grid-cols-1 gap-4 md:grid-cols-5">
            <div className="space-y-4 md:col-span-3">
              {hasSuggestions && (
                <Button type="button" variant="outline" size="sm" className="gap-2" onClick={applySuggestions}>
                  <Sparkles className="h-4 w-4" /> Use my profile details
                  <FieldHelp id="deliverables.brand_kit.use_profile" />
                </Button>
              )}
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <Label className="text-xs">Brand name</Label>
                  <Input value={kit.name} onChange={(e) => patch({ name: e.target.value })} placeholder={suggestions.name?.value || 'Acme'} />
                </div>
                <div>
                  <Label className="text-xs">Tagline</Label>
                  <Input value={kit.tagline} onChange={(e) => patch({ tagline: e.target.value })} />
                </div>
              </div>

              <div className="rounded-md border p-3">
                <Label className="flex items-center text-xs">
                  Logo <FieldHelp id="deliverables.brand_kit.logo" />
                </Label>
                <div className="mt-2 flex flex-wrap items-center gap-2">
                  <input ref={logoInput} type="file" accept="image/png,image/jpeg" className="hidden" onChange={pickFile(logo.upload)} />
                  <Button type="button" size="sm" variant="outline" className="gap-1.5" disabled={logo.busy} onClick={() => logoInput.current?.click()}>
                    {logo.busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <ImagePlus className="h-3.5 w-3.5" />}
                    {kit.logo_path ? 'Replace logo' : 'Upload logo (PNG/JPEG)'}
                  </Button>
                  {kit.logo_path && (
                    <Button type="button" size="sm" variant="ghost" className="gap-1.5 text-destructive" disabled={logo.busy} onClick={logo.remove}>
                      <Trash2 className="h-3.5 w-3.5" /> Remove
                    </Button>
                  )}
                </div>
                {!kit.logo_path && (
                  <div className="mt-2">
                    <Label className="text-xs text-muted-foreground">…or a public image URL</Label>
                    <Input value={kit.logo_url} onChange={(e) => patch({ logo_url: e.target.value })} placeholder="https://…/logo.png" />
                  </div>
                )}
              </div>

              <div className="rounded-md border p-3">
                <Label className="flex items-center text-xs">
                  Logo mark (square) <FieldHelp id="deliverables.brand_kit.logo_mark" />
                </Label>
                <div className="mt-2 flex flex-wrap items-center gap-2">
                  <input
                    ref={markInput}
                    type="file"
                    accept="image/png,image/jpeg"
                    className="hidden"
                    aria-label="Logo mark file"
                    onChange={pickFile(mark.upload)}
                  />
                  <Button type="button" size="sm" variant="outline" className="gap-1.5" disabled={mark.busy} onClick={() => markInput.current?.click()}>
                    {mark.busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <ImagePlus className="h-3.5 w-3.5" />}
                    {kit.logo_mark_path ? 'Replace logo mark' : 'Upload logo mark (square PNG/JPEG)'}
                  </Button>
                  {kit.logo_mark_path && (
                    <Button type="button" size="sm" variant="ghost" className="gap-1.5 text-destructive" disabled={mark.busy} onClick={mark.remove}>
                      <Trash2 className="h-3.5 w-3.5" /> Remove
                    </Button>
                  )}
                </div>
                {!kit.logo_mark_path && (
                  <div className="mt-2">
                    <Label htmlFor="brand-logo-mark-url" className="text-xs text-muted-foreground">…or a public image URL</Label>
                    <Input
                      id="brand-logo-mark-url"
                      value={kit.logo_mark_url}
                      onChange={(e) => patch({ logo_mark_url: e.target.value })}
                      placeholder="https://…/mark.png"
                    />
                  </div>
                )}
              </div>

              <div className="grid grid-cols-2 gap-3">
                <ColorField label="Primary (headings)" value={kit.primary_color} onChange={(v) => patch({ primary_color: v })} />
                <ColorField label="Accent (rules)" value={kit.accent_color} onChange={(v) => patch({ accent_color: v })} />
                <ColorField label="Secondary (borders)" value={kit.secondary_color} onChange={(v) => patch({ secondary_color: v })} />
                <ColorField label="Body text" value={kit.text_color} onChange={(v) => patch({ text_color: v })} />
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <Label htmlFor="brand-body-font" className="text-xs">Body font</Label>
                  <Input
                    id="brand-body-font"
                    value={kit.font_family}
                    onChange={(e) => patch({ font_family: e.target.value })}
                    placeholder="Inter, system-ui, sans-serif"
                  />
                </div>
                <div>
                  <div className="flex items-center">
                    <Label htmlFor="brand-heading-font" className="text-xs">Heading font</Label>
                    <FieldHelp id="deliverables.brand_kit.heading_font" />
                  </div>
                  <Input
                    id="brand-heading-font"
                    value={kit.heading_font}
                    onChange={(e) => patch({ heading_font: e.target.value })}
                    placeholder="Same as the body font"
                  />
                </div>
              </div>
              <BrandKitFonts fonts={kit.font_files} onChange={(font_files) => patch({ font_files })} />
              <div className="rounded-md border p-3">
                <p className="mb-2 flex items-center text-xs font-medium text-muted-foreground">
                  Company contact — fills {'{{company.*}}'} <FieldHelp id="deliverables.brand_kit.company" />
                </p>
                <div className="grid grid-cols-2 gap-3">
                  <Input placeholder="Company name" value={kit.company.name} onChange={(e) => patchCompany({ name: e.target.value })} />
                  <Input placeholder="Website" value={kit.company.website} onChange={(e) => patchCompany({ website: e.target.value })} />
                  <Input placeholder="Address" value={kit.company.address} onChange={(e) => patchCompany({ address: e.target.value })} />
                  <Input placeholder="Email" value={kit.company.email} onChange={(e) => patchCompany({ email: e.target.value })} />
                  <Input placeholder="Phone" value={kit.company.phone} onChange={(e) => patchCompany({ phone: e.target.value })} />
                </div>
              </div>
              <BrandKitSocial
                key={loads}
                handles={kit.social_handles}
                voice={kit.voice}
                onHandlesChange={(social_handles) => patch({ social_handles })}
                onVoiceChange={(voice) => patch({ voice })}
              />
            </div>
            <div className="md:col-span-2">
              <Label className="text-xs text-muted-foreground">How it renders</Label>
              <div className="mt-1">
                <KitPreview kit={kit} logoUrl={logo.objectUrl} markUrl={mark.objectUrl} />
              </div>
            </div>
          </div>
        )}

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>Cancel</Button>
          <Button onClick={save} disabled={saving || !kit || !!voiceProblem}>{saving ? 'Saving…' : 'Save'}</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
