'use client'

/**
 * PRD-255 US-007 — the logo's variants on the Brand kit page (FR-9): the logo for dark
 * backgrounds and the one-colour logo, each uploaded by the owner (never generated or
 * altered), shown on the ground it is made for.
 */
import type { BrandKit } from '@/components/documents/blocks/types'
import type { BrandImage } from '@/components/documents/blocks/useBrandImage'
import { BrandImageField } from './brand-kit-images'
import { SectionTitle } from './brand-kit-inputs'

interface VariantProps {
  title: string
  image: BrandImage
  stored: boolean
  ground: string | undefined
  canEdit: boolean
}

function Variant({ title, image, stored, ground, canEdit }: VariantProps) {
  return (
    <div className="min-w-0 space-y-2">
      <BrandImageField
        title={title} image={image} stored={stored} disabled={!canEdit}
        uploadLabel={`Upload ${title.toLowerCase()} (PNG/JPEG)`} replaceLabel={`Replace ${title.toLowerCase()}`} fileLabel={`${title} file`}
      />
      {image.objectUrl && (
        <div className="flex h-16 items-center justify-center rounded-md border" style={{ background: ground }}>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={image.objectUrl} alt={title} className="max-h-10 w-auto object-contain" />
        </div>
      )}
    </div>
  )
}

interface BrandKitLogoVariantsProps {
  kit: BrandKit
  dark: BrandImage
  mono: BrandImage
  canEdit: boolean
}

export function BrandKitLogoVariants({ kit, dark, mono, canEdit }: BrandKitLogoVariantsProps) {
  if (kit.logo_dark_path === undefined) return null
  return (
    <section aria-label="Logo variants">
      <SectionTitle title="Logo variants">Upload your own versions; the platform never makes or changes a logo.</SectionTitle>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <Variant title="Logo for dark backgrounds" image={dark} stored={!!kit.logo_dark_path} ground={kit.palette?.heading} canEdit={canEdit} />
        <Variant title="One-colour logo" image={mono} stored={!!kit.logo_mono_path} ground={kit.palette?.paper} canEdit={canEdit} />
      </div>
    </section>
  )
}
