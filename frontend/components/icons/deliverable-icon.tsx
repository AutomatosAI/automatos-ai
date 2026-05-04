/**
 * DeliverableIcon — Automatos Deliverable Icon System v1
 * =======================================================
 *
 * 7 deliverable types × 3 sizes = 21 icons.
 *
 *   - hero  (200×200) — dark canvas with radial accent glow, baked-in color
 *   - row   (24×24)   — monochrome, currentColor (color via parent text-{x} class)
 *   - badge (16×16)   — monochrome, currentColor, redrawn for clarity at small size
 *
 * Usage:
 *   <DeliverableIcon type="report" size="row" />
 *   <span className="text-info"><DeliverableIcon type="report" size="row" /></span>
 *   <DeliverableIcon type="slide" size="hero" />
 */

import * as React from 'react'

// ── Accent palette — drives Hero glow + canonical text-{x} class for Row/Badge

export const DELIVERABLE_ACCENTS = {
  report:      { hex: '#60a5fa', tw: 'text-info',    label: 'Reports'      },
  image:       { hex: '#c084fc', tw: 'text-agent',  label: 'Images'       },
  document:    { hex: '#cbd5e1', tw: 'text-slate-300',   label: 'Documents'    },
  code:        { hex: '#34d399', tw: 'text-success', label: 'Code'         },
  slide:       { hex: '#fb923c', tw: 'text-orange-400',  label: 'Slides'       },
  spreadsheet: { hex: '#4ade80', tw: 'text-success',   label: 'Spreadsheets' },
  blog_post:   { hex: '#22d3ee', tw: 'text-cyan-400',    label: 'Blog Posts'   },
} as const

export type DeliverableType = keyof typeof DELIVERABLE_ACCENTS
export type DeliverableSize = 'hero' | 'row' | 'badge'

export const DELIVERABLE_TYPES: ReadonlyArray<DeliverableType> = [
  'report', 'image', 'document', 'code', 'slide', 'spreadsheet', 'blog_post',
] as const

// ── Hero canvas wrapper — 200×200 dark with radial gradient toward accent

interface HeroCanvasProps {
  accent: string
  gradientId: string
  children: React.ReactNode
  width?: number
  height?: number
}

function HeroCanvas({ accent, gradientId, children, width, height }: HeroCanvasProps) {
  return (
    <svg
      viewBox="0 0 200 200"
      width={width ?? 200}
      height={height ?? 200}
      xmlns="http://www.w3.org/2000/svg"
    >
      <defs>
        <radialGradient id={gradientId} cx="50%" cy="35%" r="70%">
          <stop offset="0%" stopColor={accent} stopOpacity="0.18" />
          <stop offset="55%" stopColor={accent} stopOpacity="0.04" />
          <stop offset="100%" stopColor="#0a0d14" stopOpacity="0" />
        </radialGradient>
      </defs>
      <rect width="200" height="200" rx="14" fill="#0a0d14" />
      <rect width="200" height="200" rx="14" fill={`url(#${gradientId})`} />
      {children}
    </svg>
  )
}

// ── Hero icons (200×200) — multi-element, dark canvas + accent glow

interface HeroProps {
  width?: number
  height?: number
  uid: string
}

function ReportHero({ width, height, uid }: HeroProps) {
  const a = DELIVERABLE_ACCENTS.report.hex
  return (
    <HeroCanvas accent={a} gradientId={`g-report-${uid}`} width={width} height={height}>
      <rect x="44" y="40" width="112" height="120" rx="6"
        fill="none" stroke="#1f2733" strokeWidth="1.5" />
      <line x1="56" y1="132" x2="144" y2="132" stroke="#1f2733" strokeWidth="1.25" />
      <rect x="60" y="108" width="14" height="24" rx="1.5" fill={a} fillOpacity="0.22" stroke={a} strokeWidth="1.5" />
      <rect x="80" y="92"  width="14" height="40" rx="1.5" fill={a} fillOpacity="0.32" stroke={a} strokeWidth="1.5" />
      <rect x="100" y="76" width="14" height="56" rx="1.5" fill={a} fillOpacity="0.42" stroke={a} strokeWidth="1.5" />
      <rect x="120" y="60" width="14" height="72" rx="1.5" fill={a} fillOpacity="0.55" stroke={a} strokeWidth="1.5" />
      <polyline points="67,100 87,84 107,68 127,52" fill="none" stroke={a} strokeWidth="2"
        strokeLinecap="round" strokeLinejoin="round" />
      <circle cx="127" cy="52" r="3" fill={a} />
      <line x1="56" y1="52" x2="64" y2="52" stroke={a} strokeWidth="2" strokeLinecap="round" />
    </HeroCanvas>
  )
}

function ImageHero({ width, height, uid }: HeroProps) {
  const a = DELIVERABLE_ACCENTS.image.hex
  return (
    <HeroCanvas accent={a} gradientId={`g-image-${uid}`} width={width} height={height}>
      <rect x="40" y="44" width="120" height="112" rx="6"
        fill="none" stroke="#1f2733" strokeWidth="1.5" />
      <circle cx="124" cy="76" r="9" fill={a} fillOpacity="0.45" stroke={a} strokeWidth="1.5" />
      <path d="M48,140 L82,96 L116,140 Z"
        fill={a} fillOpacity="0.18" stroke={a} strokeWidth="1.5" strokeLinejoin="round" />
      <path d="M76,150 L108,108 L152,150 Z"
        fill={a} fillOpacity="0.32" stroke={a} strokeWidth="1.5" strokeLinejoin="round" />
      <line x1="48" y1="150" x2="152" y2="150" stroke="#1f2733" strokeWidth="1.25" />
    </HeroCanvas>
  )
}

function DocumentHero({ width, height, uid }: HeroProps) {
  const a = DELIVERABLE_ACCENTS.document.hex
  return (
    <HeroCanvas accent={a} gradientId={`g-document-${uid}`} width={width} height={height}>
      <path d="M58,32 L130,32 L150,52 L150,168 L58,168 Z"
        fill="none" stroke={a} strokeWidth="1.75" strokeLinejoin="round" />
      <path d="M130,32 L130,52 L150,52"
        fill={a} fillOpacity="0.14" stroke={a} strokeWidth="1.5" strokeLinejoin="round" />
      <line x1="74" y1="76"  x2="118" y2="76"  stroke={a} strokeWidth="2.5" strokeLinecap="round" />
      <line x1="74" y1="100" x2="134" y2="100" stroke={a} strokeOpacity="0.55" strokeWidth="1.75" strokeLinecap="round" />
      <line x1="74" y1="118" x2="134" y2="118" stroke={a} strokeOpacity="0.55" strokeWidth="1.75" strokeLinecap="round" />
      <line x1="74" y1="136" x2="110" y2="136" stroke={a} strokeOpacity="0.55" strokeWidth="1.75" strokeLinecap="round" />
    </HeroCanvas>
  )
}

function CodeHero({ width, height, uid }: HeroProps) {
  const a = DELIVERABLE_ACCENTS.code.hex
  return (
    <HeroCanvas accent={a} gradientId={`g-code-${uid}`} width={width} height={height}>
      <path d="M68,44 C58,44 58,72 50,100 C58,128 58,156 68,156"
        fill="none" stroke={a} strokeWidth="2.25" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M132,44 C142,44 142,72 150,100 C142,128 142,156 132,156"
        fill="none" stroke={a} strokeWidth="2.25" strokeLinecap="round" strokeLinejoin="round" />
      <line x1="78"  y1="84"  x2="118" y2="84"  stroke={a} strokeOpacity="0.7" strokeWidth="2" strokeLinecap="round" />
      <line x1="88"  y1="100" x2="122" y2="100" stroke={a} strokeWidth="2" strokeLinecap="round" />
      <line x1="78"  y1="116" x2="108" y2="116" stroke={a} strokeOpacity="0.7" strokeWidth="2" strokeLinecap="round" />
      <line x1="114" y1="116" x2="114" y2="124" stroke={a} strokeWidth="2" strokeLinecap="round" />
    </HeroCanvas>
  )
}

function SlideHero({ width, height, uid }: HeroProps) {
  const a = DELIVERABLE_ACCENTS.slide.hex
  return (
    <HeroCanvas accent={a} gradientId={`g-slide-${uid}`} width={width} height={height}>
      <rect x="48" y="56" width="100" height="68" rx="5"
        fill="#0a0d14" stroke={a} strokeOpacity="0.35" strokeWidth="1.5" />
      <rect x="56" y="68" width="100" height="68" rx="5"
        fill="#0a0d14" stroke={a} strokeOpacity="0.6" strokeWidth="1.5" />
      <rect x="44" y="84" width="112" height="76" rx="6"
        fill={a} fillOpacity="0.12" stroke={a} strokeWidth="1.75" />
      <rect x="56" y="98"  width="44" height="6" rx="2" fill={a} />
      <rect x="56" y="112" width="64" height="3" rx="1.5" fill={a} fillOpacity="0.55" />
      <rect x="56" y="120" width="52" height="3" rx="1.5" fill={a} fillOpacity="0.55" />
      <rect x="56" y="128" width="58" height="3" rx="1.5" fill={a} fillOpacity="0.55" />
      <circle cx="134" cy="124" r="13" fill={a} fillOpacity="0.18" stroke={a} strokeWidth="1.5" />
      <path d="M130,118 L130,130 L140,124 Z" fill={a} />
    </HeroCanvas>
  )
}

function SpreadsheetHero({ width, height, uid }: HeroProps) {
  const a = DELIVERABLE_ACCENTS.spreadsheet.hex
  const x0 = 52, y0 = 52, cell = 24
  return (
    <HeroCanvas accent={a} gradientId={`g-spreadsheet-${uid}`} width={width} height={height}>
      <rect x={x0} y={y0} width={cell * 4} height={cell * 4} rx="4"
        fill="none" stroke={a} strokeWidth="1.75" />
      <rect x={x0} y={y0} width={cell * 4} height={cell} fill={a} fillOpacity="0.15" />
      <rect x={x0 + cell * 2} y={y0 + cell * 2} width={cell} height={cell}
        fill={a} fillOpacity="0.45" />
      {[1, 2, 3].map((i) => (
        <line key={`v${i}`} x1={x0 + cell * i} y1={y0} x2={x0 + cell * i} y2={y0 + cell * 4}
          stroke={a} strokeOpacity="0.45" strokeWidth="1.25" />
      ))}
      {[1, 2, 3].map((i) => (
        <line key={`h${i}`} x1={x0} y1={y0 + cell * i} x2={x0 + cell * 4} y2={y0 + cell * i}
          stroke={a} strokeOpacity="0.45" strokeWidth="1.25" />
      ))}
      <circle cx={x0 + cell * 0.5} cy={y0 + cell * 0.5} r="2" fill={a} />
      <circle cx={x0 + cell * 1.5} cy={y0 + cell * 0.5} r="2" fill={a} />
      <circle cx={x0 + cell * 2.5} cy={y0 + cell * 0.5} r="2" fill={a} />
      <circle cx={x0 + cell * 3.5} cy={y0 + cell * 0.5} r="2" fill={a} />
    </HeroCanvas>
  )
}

function BlogPostHero({ width, height, uid }: HeroProps) {
  const a = DELIVERABLE_ACCENTS.blog_post.hex
  return (
    <HeroCanvas accent={a} gradientId={`g-blog-post-${uid}`} width={width} height={height}>
      <rect x="44" y="36" width="112" height="128" rx="6"
        fill="none" stroke={a} strokeWidth="1.75" />
      <rect x="56" y="48" width="88" height="40" rx="3"
        fill={a} fillOpacity="0.18" stroke={a} strokeOpacity="0.7" strokeWidth="1.25" />
      <circle cx="74" cy="64" r="4" fill={a} />
      <path d="M60,86 L82,72 L102,84 L122,68 L140,86"
        fill="none" stroke={a} strokeWidth="1.5" strokeLinejoin="round" />
      <rect x="56" y="100" width="60" height="6" rx="2" fill={a} />
      <line x1="56" y1="120" x2="144" y2="120" stroke={a} strokeOpacity="0.55" strokeWidth="1.75" strokeLinecap="round" />
      <line x1="56" y1="134" x2="144" y2="134" stroke={a} strokeOpacity="0.55" strokeWidth="1.75" strokeLinecap="round" />
      <line x1="56" y1="148" x2="116" y2="148" stroke={a} strokeOpacity="0.55" strokeWidth="1.75" strokeLinecap="round" />
    </HeroCanvas>
  )
}

const HERO_COMPONENTS: Record<DeliverableType, React.FC<HeroProps>> = {
  report: ReportHero,
  image: ImageHero,
  document: DocumentHero,
  code: CodeHero,
  slide: SlideHero,
  spreadsheet: SpreadsheetHero,
  blog_post: BlogPostHero,
}

// ── Row icons (24×24) — lucide-grade, currentColor

const rowSvgProps = {
  width: 24,
  height: 24,
  viewBox: '0 0 24 24',
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 1.5,
  strokeLinecap: 'round' as const,
  strokeLinejoin: 'round' as const,
  xmlns: 'http://www.w3.org/2000/svg',
}

function ReportRow() {
  return (
    <svg {...rowSvgProps}>
      <line x1="4" y1="20" x2="20" y2="20" />
      <rect x="6"  y="14" width="3" height="6"  rx="0.5" />
      <rect x="11" y="10" width="3" height="10" rx="0.5" />
      <rect x="16" y="6"  width="3" height="14" rx="0.5" />
    </svg>
  )
}

function ImageRow() {
  return (
    <svg {...rowSvgProps}>
      <rect x="3" y="4" width="18" height="16" rx="2" />
      <circle cx="8.5" cy="9.5" r="1.5" />
      <path d="M21 16l-5-5-9 9" />
    </svg>
  )
}

function DocumentRow() {
  return (
    <svg {...rowSvgProps}>
      <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" />
      <path d="M14 3v5h5" />
      <line x1="9"  y1="13" x2="15" y2="13" />
      <line x1="9"  y1="17" x2="13" y2="17" />
    </svg>
  )
}

function CodeRow() {
  return (
    <svg {...rowSvgProps}>
      <path d="M8 4c-2 0-3 1-3 3v2c0 1.5-.7 3-2 3 1.3 0 2 1.5 2 3v2c0 2 1 3 3 3" />
      <path d="M16 4c2 0 3 1 3 3v2c0 1.5.7 3 2 3-1.3 0-2 1.5-2 3v2c0 2-1 3-3 3" />
    </svg>
  )
}

function SlideRow() {
  return (
    <svg {...rowSvgProps}>
      <rect x="3" y="6" width="18" height="13" rx="2" />
      <line x1="6" y1="3" x2="18" y2="3" strokeOpacity="0.55" />
      <path d="M10 10v5l5-2.5z" fill="currentColor" stroke="none" />
    </svg>
  )
}

function SpreadsheetRow() {
  return (
    <svg {...rowSvgProps}>
      <rect x="3" y="3" width="18" height="18" rx="2" />
      <line x1="3"  y1="9"  x2="21" y2="9" />
      <line x1="3"  y1="15" x2="21" y2="15" />
      <line x1="9"  y1="3"  x2="9"  y2="21" />
      <line x1="15" y1="3"  x2="15" y2="21" />
    </svg>
  )
}

function BlogPostRow() {
  return (
    <svg {...rowSvgProps}>
      <rect x="4" y="3" width="16" height="18" rx="2" />
      <rect x="7" y="6" width="10" height="5" rx="0.75" />
      <line x1="7"  y1="14" x2="17" y2="14" />
      <line x1="7"  y1="17" x2="13" y2="17" />
    </svg>
  )
}

const ROW_COMPONENTS: Record<DeliverableType, React.FC> = {
  report: ReportRow,
  image: ImageRow,
  document: DocumentRow,
  code: CodeRow,
  slide: SlideRow,
  spreadsheet: SpreadsheetRow,
  blog_post: BlogPostRow,
}

// ── Badge icons (16×16) — redrawn for clarity at small size, currentColor

const badgeSvgProps = {
  width: 16,
  height: 16,
  viewBox: '0 0 16 16',
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 1.5,
  strokeLinecap: 'round' as const,
  strokeLinejoin: 'round' as const,
  xmlns: 'http://www.w3.org/2000/svg',
}

function ReportBadge() {
  return (
    <svg {...badgeSvgProps}>
      <line x1="2" y1="13.25" x2="14" y2="13.25" />
      <rect x="3.5"  y="9" width="2.5" height="4.25"  rx="0.5" />
      <rect x="6.75" y="6" width="2.5" height="7.25"  rx="0.5" />
      <rect x="10"   y="3" width="2.5" height="10.25" rx="0.5" />
    </svg>
  )
}

function ImageBadge() {
  return (
    <svg {...badgeSvgProps}>
      <rect x="2" y="3" width="12" height="10" rx="1.5" />
      <circle cx="5.5" cy="6.5" r="1" />
      <path d="M14 10.5l-3-3-5.5 5.5" />
    </svg>
  )
}

function DocumentBadge() {
  return (
    <svg {...badgeSvgProps}>
      <path d="M9.5 2H4.5a1.5 1.5 0 0 0-1.5 1.5v9a1.5 1.5 0 0 0 1.5 1.5h7a1.5 1.5 0 0 0 1.5-1.5V5.5z" />
      <path d="M9.5 2v3.5h3.5" />
      <line x1="5.5" y1="9"    x2="10.5" y2="9" />
      <line x1="5.5" y1="11.5" x2="9"    y2="11.5" />
    </svg>
  )
}

function CodeBadge() {
  return (
    <svg {...badgeSvgProps}>
      <path d="M5.5 3c-1.25 0-2 .6-2 1.75v1.5c0 .9-.5 1.75-1.25 1.75.75 0 1.25.85 1.25 1.75v1.5C3.5 12.4 4.25 13 5.5 13" />
      <path d="M10.5 3c1.25 0 2 .6 2 1.75v1.5c0 .9.5 1.75 1.25 1.75-.75 0-1.25.85-1.25 1.75v1.5C12.5 12.4 11.75 13 10.5 13" />
    </svg>
  )
}

function SlideBadge() {
  return (
    <svg {...badgeSvgProps}>
      <rect x="2" y="4" width="12" height="9" rx="1.5" />
      <line x1="4" y1="2" x2="12" y2="2" strokeOpacity="0.55" />
      <path d="M6.75 7v3l3-1.5z" fill="currentColor" stroke="none" />
    </svg>
  )
}

function SpreadsheetBadge() {
  return (
    <svg {...badgeSvgProps}>
      <rect x="2" y="2" width="12" height="12" rx="1.5" />
      <line x1="2"  y1="6"  x2="14" y2="6" />
      <line x1="2"  y1="10" x2="14" y2="10" />
      <line x1="6"  y1="2"  x2="6"  y2="14" />
      <line x1="10" y1="2"  x2="10" y2="14" />
    </svg>
  )
}

function BlogPostBadge() {
  return (
    <svg {...badgeSvgProps}>
      <rect x="2.5" y="2" width="11" height="12" rx="1.5" />
      <rect x="4.5" y="4" width="7"  height="3.5" rx="0.5" />
      <line x1="4.5" y1="9.5"   x2="11.5" y2="9.5" />
      <line x1="4.5" y1="11.75" x2="9"    y2="11.75" />
    </svg>
  )
}

const BADGE_COMPONENTS: Record<DeliverableType, React.FC> = {
  report: ReportBadge,
  image: ImageBadge,
  document: DocumentBadge,
  code: CodeBadge,
  slide: SlideBadge,
  spreadsheet: SpreadsheetBadge,
  blog_post: BlogPostBadge,
}

// ── Public API

export interface DeliverableIconProps {
  type: DeliverableType
  size: DeliverableSize
  className?: string
  width?: number
  height?: number
}

export function DeliverableIcon({ type, size, className, width, height }: DeliverableIconProps) {
  if (size === 'hero') {
    const Hero = HERO_COMPONENTS[type]
    return (
      <span className={className} style={{ display: 'inline-flex', lineHeight: 0 }}>
        <Hero width={width} height={height} uid={type} />
      </span>
    )
  }
  const map = size === 'row' ? ROW_COMPONENTS : BADGE_COMPONENTS
  const Glyph = map[type]
  return (
    <span className={className} style={{ display: 'inline-flex', lineHeight: 0 }}>
      <Glyph />
    </span>
  )
}

// ── Type-aware accent class helper — drop-in for "give me the right text-{x} class"

export function deliverableAccentClass(type: DeliverableType): string {
  return DELIVERABLE_ACCENTS[type].tw
}

export function deliverableLabel(type: DeliverableType): string {
  return DELIVERABLE_ACCENTS[type].label
}

// ── Type guard — narrows arbitrary string to DeliverableType (for fallback rendering)

export function isDeliverableType(value: string): value is DeliverableType {
  return value in DELIVERABLE_ACCENTS
}
