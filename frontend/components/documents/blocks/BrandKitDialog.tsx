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

// A live swatch of what the renderers will do with the palette (heading, rule, table head).
function KitPreview({ kit, logoUrl }: { kit: BrandKit; logoUrl: string | null }) {
  return (
    <div className="rounded-md border bg-white p-3 text-[#1a1a2e]" style={{ fontFamily: kit.font_family || undefined, color: kit.text_color || undefined }}>
      {logoUrl ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={logoUrl} alt="Logo" className="mb-2 h-8 w-auto object-contain" />
      ) : kit.logo_url ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={kit.logo_url} alt="Logo" className="mb-2 h-8 w-auto object-contain" />
      ) : null}
      <div className="text-base font-bold" style={{ color: kit.primary_color, borderBottom: `2px solid ${kit.accent_color}` }}>
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

// Edit the workspace brand kit (PRD-167 S4 → PRD-242 S3): logo upload, prefill from
// what the platform already knows, every colour the renderers use, and a live swatch.
export function BrandKitDialog({ open, onOpenChange, onSaved }: BrandKitDialogProps) {
  const [kit, setKit] = useState<BrandKit | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [logoUrl, setLogoUrl] = useState<string | null>(null)
  const [suggestions, setSuggestions] = useState<BrandSuggestions>({})
  const fileInput = useRef<HTMLInputElement | null>(null)

  const refreshLogo = async (hasLogo: boolean) => {
    if (!hasLogo) {
      setLogoUrl((prev) => {
        if (prev) URL.revokeObjectURL(prev)
        return null
      })
      return
    }
    try {
      const url = await templateBlocksApi.fetchLogoObjectUrl()
      setLogoUrl((prev) => {
        if (prev) URL.revokeObjectURL(prev)
        return url
      })
    } catch {
      setLogoUrl(null)
    }
  }

  useEffect(() => {
    if (!open) return
    setLoadError(null)
    templateBlocksApi
      .getBrandKit()
      .then((k) => {
        setKit(k)
        refreshLogo(!!k.logo_path)
      })
      .catch((e: any) => setLoadError(e?.message || 'Failed to load brand kit'))
    templateBlocksApi
      .getBrandSuggestions()
      .then((r) => setSuggestions(r.suggestions || {}))
      .catch(() => setSuggestions({}))
  }, [open])

  const patch = (p: Partial<BrandKit>) => setKit((k) => (k ? { ...k, ...p } : k))
  const patchCompany = (p: Partial<BrandKit['company']>) =>
    setKit((k) => (k ? { ...k, company: { ...k.company, ...p } } : k))

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

  const uploadLogo = async (file: File | undefined) => {
    if (!file) return
    setUploading(true)
    try {
      const saved = await templateBlocksApi.uploadLogo(file)
      setKit((k) => (k ? { ...k, logo_path: saved.logo_path, logo_url: saved.logo_url } : k))
      await refreshLogo(true)
      toast.success('Logo uploaded — it will appear in every document with a logo block')
    } catch (e: any) {
      toast.error(e?.message || 'Logo upload failed')
    } finally {
      setUploading(false)
      if (fileInput.current) fileInput.current.value = ''
    }
  }

  const removeLogo = async () => {
    setUploading(true)
    try {
      const saved = await templateBlocksApi.deleteLogo()
      setKit((k) => (k ? { ...k, logo_path: saved.logo_path } : k))
      await refreshLogo(false)
      toast.success('Logo removed')
    } catch (e: any) {
      toast.error(e?.message || 'Could not remove the logo')
    } finally {
      setUploading(false)
    }
  }

  const save = async () => {
    if (!kit) return
    setSaving(true)
    try {
      // logo_path is server-managed; the update route ignores it (validate_brand_kit strips it).
      const saved = await templateBlocksApi.updateBrandKit(kit)
      toast.success('Brand kit saved')
      onSaved?.(saved)
      onOpenChange(false)
    } catch (e: any) {
      toast.error(e?.message || 'Failed to save brand kit')
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
            Applied to every document rendered from a template — logo, colours, font, and the {'{{brand.*}}'} / {'{{company.*}}'} chips.
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
                  <input
                    ref={fileInput}
                    type="file"
                    accept="image/png,image/jpeg"
                    className="hidden"
                    onChange={(e) => uploadLogo(e.target.files?.[0])}
                  />
                  <Button type="button" size="sm" variant="outline" className="gap-1.5" disabled={uploading} onClick={() => fileInput.current?.click()}>
                    {uploading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <ImagePlus className="h-3.5 w-3.5" />}
                    {kit.logo_path ? 'Replace logo' : 'Upload logo (PNG/JPEG)'}
                  </Button>
                  {kit.logo_path && (
                    <Button type="button" size="sm" variant="ghost" className="gap-1.5 text-destructive" disabled={uploading} onClick={removeLogo}>
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

              <div className="grid grid-cols-2 gap-3">
                <ColorField label="Primary (headings)" value={kit.primary_color} onChange={(v) => patch({ primary_color: v })} />
                <ColorField label="Accent (rules)" value={kit.accent_color} onChange={(v) => patch({ accent_color: v })} />
                <ColorField label="Secondary (borders)" value={kit.secondary_color} onChange={(v) => patch({ secondary_color: v })} />
                <ColorField label="Body text" value={kit.text_color} onChange={(v) => patch({ text_color: v })} />
              </div>
              <div>
                <Label className="text-xs">Font family</Label>
                <Input value={kit.font_family} onChange={(e) => patch({ font_family: e.target.value })} placeholder="Inter, system-ui, sans-serif" />
              </div>
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
            </div>
            <div className="md:col-span-2">
              <Label className="text-xs text-muted-foreground">How it renders</Label>
              <div className="mt-1">
                <KitPreview kit={kit} logoUrl={logoUrl} />
              </div>
            </div>
          </div>
        )}

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>Cancel</Button>
          <Button onClick={save} disabled={saving || !kit}>{saving ? 'Saving…' : 'Save'}</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
