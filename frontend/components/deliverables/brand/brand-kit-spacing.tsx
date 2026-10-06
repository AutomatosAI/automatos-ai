'use client'

/**
 * PRD-255 US-007 — spacing and the logo's rules on the Brand kit page: the spacing unit, the
 * page margin, the letterhead logo's height, its clear space and its least size, with a page
 * drawn to scale (A4) showing the margin, the logo and its clear space.
 */
import type { BrandKit } from '@/components/documents/blocks/types'
import { NumberField, SectionTitle } from './brand-kit-inputs'

const A4_WIDTH_MM = 210
const A4_HEIGHT_MM = 297
const PREVIEW_WIDTH_PX = 168
const PX_PER_MM = PREVIEW_WIDTH_PX / A4_WIDTH_MM
const MM_PER_PT = 25.4 / 72
// The grid stripes in the preview: one line every this many spacing units.
const GRID_UNITS_PER_LINE = 4
const MM_STEP = 0.5
const PT_STEP = 0.5
const CLEAR_SPACE_STEP = 0.1
type LogoRules = NonNullable<BrandKit['logo_rules']>

interface BrandKitSpacingProps {
  kit: BrandKit
  logoUrl: string | null
  patch: (p: Partial<BrandKit>) => void
}

function PagePreview({ kit, logoUrl, rules }: { kit: BrandKit; logoUrl: string | null; rules: LogoRules }) {
  const margin = (kit.page_margin_mm ?? 0) * PX_PER_MM
  const logoHeight = rules.letterhead_mm * PX_PER_MM
  const clear = rules.clear_space * logoHeight
  const gridPx = (kit.spacing_unit_pt ?? 0) * GRID_UNITS_PER_LINE * MM_PER_PT * PX_PER_MM
  const rule = kit.palette?.rule
  return (
    <div
      aria-label="Page preview" role="img" className="relative shrink-0 border shadow-sm"
      style={{ width: PREVIEW_WIDTH_PX, height: A4_HEIGHT_MM * PX_PER_MM, background: kit.palette?.paper }}
    >
      <div
        className="absolute border border-dashed" data-testid="preview-margin"
        style={{
          top: margin, right: margin, bottom: margin, left: margin, borderColor: rule,
          backgroundImage: gridPx > 0 ? `repeating-linear-gradient(to bottom, transparent 0 ${gridPx - 1}px, ${rule ?? 'transparent'} ${gridPx - 1}px ${gridPx}px)` : undefined,
        }}
      >
        <div className="inline-block border border-dotted" data-testid="preview-clear-space" style={{ padding: clear, borderColor: kit.palette?.accent }}>
          {logoUrl ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={logoUrl} alt="Letterhead logo" style={{ height: logoHeight }} className="w-auto object-contain" />
          ) : (
            <div style={{ height: logoHeight, width: logoHeight * 2, background: kit.palette?.surface_2 }} />
          )}
        </div>
      </div>
    </div>
  )
}

export function BrandKitSpacing({ kit, logoUrl, patch }: BrandKitSpacingProps) {
  const rules = kit.logo_rules
  if (!rules) return null
  const setRule = (change: Partial<LogoRules>) => patch({ logo_rules: { ...rules, ...change } })
  return (
    <section aria-label="Spacing and logo">
      <SectionTitle title="Spacing and logo">The grid every gap is a multiple of, the page margin, and how the letterhead places the logo.</SectionTitle>
      <div className="flex flex-col gap-4 sm:flex-row">
        <div className="grid flex-1 grid-cols-2 content-start gap-3">
          <NumberField id="brand-spacing-unit" label="Spacing unit" unit="pt" step={PT_STEP} value={kit.spacing_unit_pt} onChange={(spacing_unit_pt) => patch({ spacing_unit_pt })} />
          <NumberField id="brand-page-margin" label="Page margin" unit="mm" step={MM_STEP} value={kit.page_margin_mm} onChange={(page_margin_mm) => patch({ page_margin_mm })} />
          <NumberField id="brand-logo-letterhead" label="Letterhead logo height" unit="mm" step={MM_STEP} value={rules.letterhead_mm} onChange={(letterhead_mm) => setRule({ letterhead_mm })} />
          <NumberField id="brand-logo-clear" label="Clear space" unit="logo heights" step={CLEAR_SPACE_STEP} value={rules.clear_space} onChange={(clear_space) => setRule({ clear_space })} />
          <NumberField id="brand-logo-min" label="Least logo size" unit="mm" step={MM_STEP} value={rules.min_mm} onChange={(min_mm) => setRule({ min_mm })} />
        </div>
        <PagePreview kit={kit} logoUrl={logoUrl} rules={rules} />
      </div>
    </section>
  )
}
