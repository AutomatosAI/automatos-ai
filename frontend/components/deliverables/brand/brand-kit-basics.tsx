'use client'

/**
 * PRD-251B US-B301 — the Brand kit tab's basics card (the mockup's): the brand's name and
 * tagline, the logo and the logo mark, its colours and fonts, the company contact, the
 * social handles and the voice. What the BrandKitDialog held, on the tab; owners and admins
 * (workspace:manage) change it, everyone reads it. F371: the "How it renders" swatch beside it
 * is gone; the brand board above the card (brand-board-preview.tsx) is the kit's one preview.
 */
import { Loader2, Sparkles } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { FieldHelp } from '@/components/ui/help-tooltip'
import { BrandKitSocial } from '@/components/documents/blocks/BrandKitSocial'
import type { BrandKitForm } from './use-brand-kit-form'
import { BrandColours, BrandFonts, CompanyFields } from './brand-kit-fields'
import { BrandImageField } from './brand-kit-images'

interface BrandKitBasicsProps {
  form: BrandKitForm
  canEdit: boolean
}

export function BrandKitBasics({ form, canEdit }: BrandKitBasicsProps) {
  const { kit, loadError, suggestions, logo, mark, patch, patchCompany } = form
  if (loadError) return <p className="py-6 text-center text-sm text-destructive">{loadError}</p>
  if (!kit) {
    return (
      <p className="flex items-center justify-center gap-2 py-8 text-sm text-muted-foreground">
        <Loader2 className="h-4 w-4 animate-spin" aria-hidden /> Loading the brand kit…
      </p>
    )
  }
  return (
    <section aria-label="The basics" className="rounded-xl border bg-card p-4">
      <fieldset disabled={!canEdit} className="min-w-0 space-y-4">
        {canEdit && form.hasSuggestions && (
          <Button type="button" variant="outline" size="sm" className="gap-2" onClick={form.applySuggestions}>
            <Sparkles className="h-4 w-4" aria-hidden /> Use my profile details
            <FieldHelp id="deliverables.brand_kit.use_profile" />
          </Button>
        )}
        <div className="grid grid-cols-2 gap-3">
          <div>
            <Label htmlFor="brand-name" className="text-xs">Brand name</Label>
            <Input id="brand-name" value={kit.name} onChange={(e) => patch({ name: e.target.value })} placeholder={suggestions.name?.value || 'Acme'} />
          </div>
          <div>
            <Label htmlFor="brand-tagline" className="text-xs">Tagline</Label>
            <Input id="brand-tagline" value={kit.tagline} onChange={(e) => patch({ tagline: e.target.value })} />
          </div>
        </div>
        <BrandImageField
          title="Logo" help="deliverables.brand_kit.logo" image={logo} stored={!!kit.logo_path} url={kit.logo_url}
          uploadLabel="Upload logo (PNG/JPEG)" replaceLabel="Replace logo" fileLabel="Logo file" urlId="brand-logo-url"
          urlPlaceholder="https://…/logo.png" disabled={!canEdit} onUrl={(logo_url) => patch({ logo_url })}
        />
        <BrandImageField
          title="Logo mark (square)" help="deliverables.brand_kit.logo_mark" image={mark} stored={!!kit.logo_mark_path}
          url={kit.logo_mark_url} uploadLabel="Upload logo mark (square PNG/JPEG)" replaceLabel="Replace logo mark"
          fileLabel="Logo mark file" urlId="brand-logo-mark-url" urlPlaceholder="https://…/mark.png" disabled={!canEdit}
          onUrl={(logo_mark_url) => patch({ logo_mark_url })}
        />
        <BrandColours kit={kit} patch={patch} />
        <BrandFonts kit={kit} patch={patch} patchStored={form.patchStored} />
        <CompanyFields company={kit.company} onChange={patchCompany} />
        <BrandKitSocial
          key={form.loads}
          handles={kit.social_handles}
          voice={kit.voice}
          onHandlesChange={(social_handles) => patch({ social_handles })}
          onVoiceChange={(voice) => patch({ voice })}
        />
        {canEdit && (
          <div className="flex justify-end">
            <Button type="button" onClick={form.save} disabled={form.saving || !!form.voiceProblem}>
              {form.saving ? 'Saving…' : 'Save'}
            </Button>
          </div>
        )}
      </fieldset>
    </section>
  )
}
