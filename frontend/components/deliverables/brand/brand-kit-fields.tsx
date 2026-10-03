'use client'

/**
 * PRD-251B US-B301 — the Brand kit tab's colour, font and company fields: every colour the
 * renderers use, the body and heading fonts with the uploaded font files, and the contact
 * details that fill {{company.*}}.
 */
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { FieldHelp } from '@/components/ui/help-tooltip'
import { BrandKitFonts } from '@/components/documents/blocks/BrandKitFonts'
import type { BrandKit } from '@/components/documents/blocks/types'

const HEX = /^#([0-9a-fA-F]{6})$/
const FALLBACK_SWATCH = '#1a1a2e'

function ColorField({ label, value, onChange }: { label: string; value: string; onChange: (v: string) => void }) {
  return (
    <div>
      <Label className="text-xs">{label}</Label>
      <div className="flex items-center gap-2">
        <input
          type="color"
          value={HEX.test(value) ? value : FALLBACK_SWATCH}
          onChange={(e) => onChange(e.target.value)}
          className="h-9 w-10 cursor-pointer rounded border bg-transparent"
          aria-label={label}
        />
        <Input value={value} onChange={(e) => onChange(e.target.value)} className="font-mono text-sm" />
      </div>
    </div>
  )
}

interface KitFieldsProps {
  kit: BrandKit
  patch: (p: Partial<BrandKit>) => void
}

export function BrandColours({ kit, patch }: KitFieldsProps) {
  return (
    <div className="grid grid-cols-2 gap-3">
      <ColorField label="Primary (headings)" value={kit.primary_color} onChange={(v) => patch({ primary_color: v })} />
      <ColorField label="Accent (rules)" value={kit.accent_color} onChange={(v) => patch({ accent_color: v })} />
      <ColorField label="Secondary (borders)" value={kit.secondary_color} onChange={(v) => patch({ secondary_color: v })} />
      <ColorField label="Body text" value={kit.text_color} onChange={(v) => patch({ text_color: v })} />
    </div>
  )
}

export function BrandFonts({ kit, patch }: KitFieldsProps) {
  return (
    <>
      <div className="grid grid-cols-2 gap-3">
        <div>
          <Label htmlFor="brand-body-font" className="text-xs">Body font</Label>
          <Input id="brand-body-font" value={kit.font_family} onChange={(e) => patch({ font_family: e.target.value })} placeholder="Inter, system-ui, sans-serif" />
        </div>
        <div>
          <div className="flex items-center">
            <Label htmlFor="brand-heading-font" className="text-xs">Heading font</Label>
            <FieldHelp id="deliverables.brand_kit.heading_font" />
          </div>
          <Input id="brand-heading-font" value={kit.heading_font} onChange={(e) => patch({ heading_font: e.target.value })} placeholder="Same as the body font" />
        </div>
      </div>
      <BrandKitFonts fonts={kit.font_files} onChange={(font_files) => patch({ font_files })} />
    </>
  )
}

interface CompanyFieldsProps {
  company: BrandKit['company']
  onChange: (p: Partial<BrandKit['company']>) => void
}

export function CompanyFields({ company, onChange }: CompanyFieldsProps) {
  return (
    <div className="rounded-md border p-3">
      <p className="mb-2 flex items-center text-xs font-medium text-muted-foreground">
        Company contact — fills {'{{company.*}}'} <FieldHelp id="deliverables.brand_kit.company" />
      </p>
      <div className="grid grid-cols-2 gap-3">
        <Input placeholder="Company name" value={company.name} onChange={(e) => onChange({ name: e.target.value })} />
        <Input placeholder="Website" value={company.website} onChange={(e) => onChange({ website: e.target.value })} />
        <Input placeholder="Address" value={company.address} onChange={(e) => onChange({ address: e.target.value })} />
        <Input placeholder="Email" value={company.email} onChange={(e) => onChange({ email: e.target.value })} />
        <Input placeholder="Phone" value={company.phone} onChange={(e) => onChange({ phone: e.target.value })} />
      </div>
    </div>
  )
}
