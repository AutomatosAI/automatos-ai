'use client'

/** PRD-251B US-B301 — a live swatch of what the renderers do with the kit (heading, rule, table head). */
import type { BrandKit } from '@/components/documents/blocks/types'

interface BrandKitPreviewProps {
  kit: BrandKit
  logoUrl: string | null
  markUrl: string | null
}

export function BrandKitPreview({ kit, logoUrl, markUrl }: BrandKitPreviewProps) {
  const logo = logoUrl || kit.logo_url || null
  const mark = markUrl || kit.logo_mark_url || null
  return (
    <div className="rounded-md border bg-white p-3 text-[#1a1a2e]" style={{ fontFamily: kit.font_family || undefined, color: kit.text_color || undefined }}>
      {logo && (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={logo} alt="Logo" className="mb-2 h-8 w-auto object-contain" />
      )}
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
