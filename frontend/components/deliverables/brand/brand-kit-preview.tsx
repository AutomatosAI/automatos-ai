'use client'

/**
 * PRD-251B US-B301 — a live swatch of what the renderers do with the kit. PRD-255: it reads
 * the colour roles as the renderers do (blocks/design_tokens.py): the heading in `heading`,
 * the title rule in the accent, the table header in `surface_2` (or, under a bold accent, the
 * accent) with white text where white reads, else the heading colour.
 */
import type { BrandKit } from '@/components/documents/blocks/types'
import { contrastRatio, TEXT_MIN_CONTRAST } from './contrast'

const WHITE = '#ffffff'

interface BrandKitPreviewProps {
  kit: BrandKit
  logoUrl: string | null
  markUrl: string | null
}

/** The table header's fill and text for ``kit``, as the document renderers paint it. */
export function headerColours(kit: BrandKit): { fill?: string; text?: string } {
  const roles = kit.palette ?? {}
  const fill = kit.accent_use === 'bold' ? roles.accent : roles.surface_2
  const whiteReads = (contrastRatio(fill, WHITE) ?? 0) >= TEXT_MIN_CONTRAST
  return { fill, text: whiteReads ? WHITE : roles.heading }
}

export function BrandKitPreview({ kit, logoUrl, markUrl }: BrandKitPreviewProps) {
  const logo = logoUrl || kit.logo_url || null
  const mark = markUrl || kit.logo_mark_url || null
  const roles = kit.palette ?? {}
  const header = headerColours(kit)
  return (
    <div className="rounded-md border p-3" style={{ fontFamily: kit.font_family || undefined, color: roles.ink, background: roles.paper }}>
      {logo && (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={logo} alt="Logo" className="mb-2 h-8 w-auto object-contain" />
      )}
      <div
        className="flex items-center gap-2 text-base font-bold"
        style={{ color: roles.heading, borderBottom: `2px solid ${roles.accent}`, fontFamily: kit.heading_font || undefined }}
      >
        {mark && (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={mark} alt="Logo mark" className="h-5 w-5 object-contain" />
        )}
        {kit.name || 'Your brand'}
      </div>
      <div className="mt-1 text-[11px]" style={{ color: roles.muted }}>{kit.tagline || 'Tagline'} · {kit.company.website || 'website'}</div>
      <div className="mt-2 grid grid-cols-2 text-[10px]">
        <div className="px-2 py-1 font-semibold" data-testid="preview-table-header" style={{ background: header.fill, color: header.text }}>Table header</div>
        <div className="border px-2 py-1" style={{ borderColor: roles.rule, background: roles.surface }}>Cell</div>
      </div>
    </div>
  )
}
