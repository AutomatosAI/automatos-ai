'use client'

/**
 * PRD-255 US-007 — the kit's design system on the Brand kit tab, under the basics: the
 * colour roles, the type scale, spacing and the logo's rules, the logo's variants and the
 * locale. One kit and one Save: this card's Save sends the whole kit, as the basics' does.
 */
import { Button } from '@/components/ui/button'
import type { BrandKitForm } from './use-brand-kit-form'
import { BrandKitColours } from './brand-kit-colours'
import { BrandKitLocale } from './brand-kit-locale'
import { BrandKitLogoVariants } from './brand-kit-logo-variants'
import { BrandKitSpacing } from './brand-kit-spacing'
import { BrandKitType } from './brand-kit-type'

interface BrandKitDesignProps {
  form: BrandKitForm
  canEdit: boolean
}

export function BrandKitDesign({ form, canEdit }: BrandKitDesignProps) {
  const { kit, patch } = form
  // A backend without the colour roles has none of the design system to show.
  if (!kit?.palette) return null
  return (
    <section aria-label="The design system" className="rounded-xl border bg-card p-4">
      <fieldset disabled={!canEdit} className="min-w-0 space-y-6">
        <BrandKitColours kit={kit} edits={form.palette} canEdit={canEdit} patch={patch} />
        <BrandKitType kit={kit} patch={patch} />
        <BrandKitSpacing kit={kit} logoUrl={form.logo.objectUrl || kit.logo_url || null} patch={patch} />
        <BrandKitLogoVariants kit={kit} dark={form.dark} mono={form.mono} canEdit={canEdit} />
        <BrandKitLocale kit={kit} patch={patch} />
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
