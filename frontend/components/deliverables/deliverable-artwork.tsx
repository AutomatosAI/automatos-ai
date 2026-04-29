'use client'

/**
 * DeliverableArtwork
 * ==================
 *
 * Static SVG artwork for deliverable cards. Each artifact_type gets a unique
 * illustrated tile with type-specific glyph + accent colour.
 *
 *  - Full-bleed: 100% width/height SVG (parent enforces aspect-square)
 *  - Dark canvas + radial accent glow + per-type illustration
 *  - Static — no animations (a wall of these tiles needs to feel calm)
 *  - Deterministic IDs via useId() so multiple cards on a page don't collide
 *
 * Canonical types (the 7 design-system deliverables) match the row/badge icon
 * artwork in `@/components/icons/deliverable-icon`. Non-canonical types
 * (archive, audio, video) keep simple fallback artwork so the system covers
 * every artifact_type the backend produces.
 */

import { memo, useId } from 'react'

import { DELIVERABLE_ACCENTS, type DeliverableType } from '@/components/icons/deliverable-icon'

type FallbackType = 'archive' | 'audio' | 'video' | 'default'
type ArtworkType = DeliverableType | FallbackType

const FALLBACK_ACCENT: Record<FallbackType, string> = {
  archive: '#fbbf24',
  audio: '#f472b6',
  video: '#f87171',
  default: '#94a3b8',
}

const CANVAS_BG = '#0a0d14'

interface DeliverableArtworkProps {
  type: string
  className?: string
}

function isCanonical(t: string): t is DeliverableType {
  return t in DELIVERABLE_ACCENTS
}

function normalizeType(t: string): ArtworkType {
  if (isCanonical(t)) return t
  if (t === 'archive' || t === 'audio' || t === 'video') return t
  return 'default'
}

function accentFor(type: ArtworkType): string {
  if (isCanonical(type)) return DELIVERABLE_ACCENTS[type].hex
  return FALLBACK_ACCENT[type]
}

// ─── Canonical artwork (matches DeliverableIcon hero designs) ───────

function ReportArt({ fg }: { fg: string }) {
  return (
    <g>
      <rect x="44" y="40" width="112" height="120" rx="6" fill="none" stroke="#1f2733" strokeWidth="1.5" />
      <line x1="56" y1="132" x2="144" y2="132" stroke="#1f2733" strokeWidth="1.25" />
      <rect x="60"  y="108" width="14" height="24" rx="1.5" fill={fg} fillOpacity="0.22" stroke={fg} strokeWidth="1.5" />
      <rect x="80"  y="92"  width="14" height="40" rx="1.5" fill={fg} fillOpacity="0.32" stroke={fg} strokeWidth="1.5" />
      <rect x="100" y="76"  width="14" height="56" rx="1.5" fill={fg} fillOpacity="0.42" stroke={fg} strokeWidth="1.5" />
      <rect x="120" y="60"  width="14" height="72" rx="1.5" fill={fg} fillOpacity="0.55" stroke={fg} strokeWidth="1.5" />
      <polyline points="67,100 87,84 107,68 127,52" fill="none" stroke={fg} strokeWidth="2"
        strokeLinecap="round" strokeLinejoin="round" />
      <circle cx="127" cy="52" r="3" fill={fg} />
      <line x1="56" y1="52" x2="64" y2="52" stroke={fg} strokeWidth="2" strokeLinecap="round" />
    </g>
  )
}

function ImageArt({ fg }: { fg: string }) {
  return (
    <g>
      <rect x="40" y="44" width="120" height="112" rx="6" fill="none" stroke="#1f2733" strokeWidth="1.5" />
      <circle cx="124" cy="76" r="9" fill={fg} fillOpacity="0.45" stroke={fg} strokeWidth="1.5" />
      <path d="M48,140 L82,96 L116,140 Z" fill={fg} fillOpacity="0.18" stroke={fg} strokeWidth="1.5" strokeLinejoin="round" />
      <path d="M76,150 L108,108 L152,150 Z" fill={fg} fillOpacity="0.32" stroke={fg} strokeWidth="1.5" strokeLinejoin="round" />
      <line x1="48" y1="150" x2="152" y2="150" stroke="#1f2733" strokeWidth="1.25" />
    </g>
  )
}

function DocumentArt({ fg }: { fg: string }) {
  return (
    <g>
      <path d="M58,32 L130,32 L150,52 L150,168 L58,168 Z"
        fill="none" stroke={fg} strokeWidth="1.75" strokeLinejoin="round" />
      <path d="M130,32 L130,52 L150,52" fill={fg} fillOpacity="0.14" stroke={fg} strokeWidth="1.5" strokeLinejoin="round" />
      <line x1="74" y1="76"  x2="118" y2="76"  stroke={fg} strokeWidth="2.5" strokeLinecap="round" />
      <line x1="74" y1="100" x2="134" y2="100" stroke={fg} strokeOpacity="0.55" strokeWidth="1.75" strokeLinecap="round" />
      <line x1="74" y1="118" x2="134" y2="118" stroke={fg} strokeOpacity="0.55" strokeWidth="1.75" strokeLinecap="round" />
      <line x1="74" y1="136" x2="110" y2="136" stroke={fg} strokeOpacity="0.55" strokeWidth="1.75" strokeLinecap="round" />
    </g>
  )
}

function CodeArt({ fg }: { fg: string }) {
  return (
    <g>
      <path d="M68,44 C58,44 58,72 50,100 C58,128 58,156 68,156"
        fill="none" stroke={fg} strokeWidth="2.25" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M132,44 C142,44 142,72 150,100 C142,128 142,156 132,156"
        fill="none" stroke={fg} strokeWidth="2.25" strokeLinecap="round" strokeLinejoin="round" />
      <line x1="78"  y1="84"  x2="118" y2="84"  stroke={fg} strokeOpacity="0.7" strokeWidth="2" strokeLinecap="round" />
      <line x1="88"  y1="100" x2="122" y2="100" stroke={fg} strokeWidth="2" strokeLinecap="round" />
      <line x1="78"  y1="116" x2="108" y2="116" stroke={fg} strokeOpacity="0.7" strokeWidth="2" strokeLinecap="round" />
      <line x1="114" y1="116" x2="114" y2="124" stroke={fg} strokeWidth="2" strokeLinecap="round" />
    </g>
  )
}

function SlideArt({ fg }: { fg: string }) {
  return (
    <g>
      <rect x="48" y="56" width="100" height="68" rx="5" fill={CANVAS_BG} stroke={fg} strokeOpacity="0.35" strokeWidth="1.5" />
      <rect x="56" y="68" width="100" height="68" rx="5" fill={CANVAS_BG} stroke={fg} strokeOpacity="0.6" strokeWidth="1.5" />
      <rect x="44" y="84" width="112" height="76" rx="6" fill={fg} fillOpacity="0.12" stroke={fg} strokeWidth="1.75" />
      <rect x="56" y="98"  width="44" height="6" rx="2" fill={fg} />
      <rect x="56" y="112" width="64" height="3" rx="1.5" fill={fg} fillOpacity="0.55" />
      <rect x="56" y="120" width="52" height="3" rx="1.5" fill={fg} fillOpacity="0.55" />
      <rect x="56" y="128" width="58" height="3" rx="1.5" fill={fg} fillOpacity="0.55" />
      <circle cx="134" cy="124" r="13" fill={fg} fillOpacity="0.18" stroke={fg} strokeWidth="1.5" />
      <path d="M130,118 L130,130 L140,124 Z" fill={fg} />
    </g>
  )
}

function SpreadsheetArt({ fg }: { fg: string }) {
  const x0 = 52, y0 = 52, cell = 24
  return (
    <g>
      <rect x={x0} y={y0} width={cell * 4} height={cell * 4} rx="4" fill="none" stroke={fg} strokeWidth="1.75" />
      <rect x={x0} y={y0} width={cell * 4} height={cell} fill={fg} fillOpacity="0.15" />
      <rect x={x0 + cell * 2} y={y0 + cell * 2} width={cell} height={cell} fill={fg} fillOpacity="0.45" />
      {[1, 2, 3].map((i) => (
        <line key={`v${i}`} x1={x0 + cell * i} y1={y0} x2={x0 + cell * i} y2={y0 + cell * 4}
          stroke={fg} strokeOpacity="0.45" strokeWidth="1.25" />
      ))}
      {[1, 2, 3].map((i) => (
        <line key={`h${i}`} x1={x0} y1={y0 + cell * i} x2={x0 + cell * 4} y2={y0 + cell * i}
          stroke={fg} strokeOpacity="0.45" strokeWidth="1.25" />
      ))}
      <circle cx={x0 + cell * 0.5} cy={y0 + cell * 0.5} r="2" fill={fg} />
      <circle cx={x0 + cell * 1.5} cy={y0 + cell * 0.5} r="2" fill={fg} />
      <circle cx={x0 + cell * 2.5} cy={y0 + cell * 0.5} r="2" fill={fg} />
      <circle cx={x0 + cell * 3.5} cy={y0 + cell * 0.5} r="2" fill={fg} />
    </g>
  )
}

function BlogPostArt({ fg }: { fg: string }) {
  return (
    <g>
      <rect x="44" y="36" width="112" height="128" rx="6" fill="none" stroke={fg} strokeWidth="1.75" />
      <rect x="56" y="48" width="88" height="40" rx="3" fill={fg} fillOpacity="0.18" stroke={fg} strokeOpacity="0.7" strokeWidth="1.25" />
      <circle cx="74" cy="64" r="4" fill={fg} />
      <path d="M60,86 L82,72 L102,84 L122,68 L140,86" fill="none" stroke={fg} strokeWidth="1.5" strokeLinejoin="round" />
      <rect x="56" y="100" width="60" height="6" rx="2" fill={fg} />
      <line x1="56" y1="120" x2="144" y2="120" stroke={fg} strokeOpacity="0.55" strokeWidth="1.75" strokeLinecap="round" />
      <line x1="56" y1="134" x2="144" y2="134" stroke={fg} strokeOpacity="0.55" strokeWidth="1.75" strokeLinecap="round" />
      <line x1="56" y1="148" x2="116" y2="148" stroke={fg} strokeOpacity="0.55" strokeWidth="1.75" strokeLinecap="round" />
    </g>
  )
}

// ─── Fallback artwork (archive, audio, video, default) ──────────────

function ArchiveArt({ fg }: { fg: string }) {
  return (
    <g>
      <rect x="55" y="80" width="90" height="70" rx="3" fill={fg} fillOpacity="0.15" stroke={fg} strokeOpacity="0.6" strokeWidth="1.5" />
      <rect x="50" y="65" width="100" height="20" rx="2" fill={fg} fillOpacity="0.25" stroke={fg} strokeOpacity="0.7" strokeWidth="1.5" />
      <line x1="100" y1="60" x2="100" y2="155" stroke={fg} strokeOpacity="0.6" strokeWidth="3" />
    </g>
  )
}

function AudioArt({ fg }: { fg: string }) {
  const bars = [
    { x: 50, base: 30 },
    { x: 65, base: 55 },
    { x: 80, base: 40 },
    { x: 95, base: 70 },
    { x: 110, base: 45 },
    { x: 125, base: 60 },
    { x: 140, base: 35 },
  ]
  return (
    <g>
      {bars.map((b) => (
        <rect key={b.x} x={b.x} y={100 - b.base / 2} width="8" height={b.base} rx="2" fill={fg} fillOpacity="0.7" />
      ))}
    </g>
  )
}

function VideoArt({ fg }: { fg: string }) {
  return (
    <g>
      <circle cx="100" cy="100" r="50" fill={fg} fillOpacity="0.15" stroke={fg} strokeOpacity="0.4" strokeWidth="1.5" />
      <polygon points="90,80 90,120 125,100" fill={fg} fillOpacity="0.85" />
    </g>
  )
}

function DefaultArt({ fg }: { fg: string }) {
  return (
    <g>
      <path d="M 75 50 L 115 50 L 135 70 L 135 150 L 75 150 Z"
        fill={fg} fillOpacity="0.1" stroke={fg} strokeOpacity="0.5" strokeWidth="1.5" />
      <path d="M 115 50 L 115 70 L 135 70" fill="none" stroke={fg} strokeOpacity="0.5" strokeWidth="1.5" />
    </g>
  )
}

// ─── Main component ─────────────────────────────────────────────────

function DeliverableArtworkImpl({ type, className }: DeliverableArtworkProps) {
  const id = useId().replace(/:/g, '')
  const norm = normalizeType(type)
  const accent = accentFor(norm)
  const gradId = `grad-${id}-${norm}`

  return (
    <svg
      viewBox="0 0 200 200"
      width="100%"
      height="100%"
      preserveAspectRatio="xMidYMid slice"
      className={className}
      role="img"
      aria-label={`${norm} preview`}
    >
      <defs>
        <radialGradient id={gradId} cx="50%" cy="35%" r="70%">
          <stop offset="0%" stopColor={accent} stopOpacity="0.18" />
          <stop offset="55%" stopColor={accent} stopOpacity="0.04" />
          <stop offset="100%" stopColor={CANVAS_BG} stopOpacity="0" />
        </radialGradient>
      </defs>

      <rect x="0" y="0" width="200" height="200" fill={CANVAS_BG} />
      <rect x="0" y="0" width="200" height="200" fill={`url(#${gradId})`} />

      {norm === 'report' && <ReportArt fg={accent} />}
      {norm === 'image' && <ImageArt fg={accent} />}
      {norm === 'document' && <DocumentArt fg={accent} />}
      {norm === 'code' && <CodeArt fg={accent} />}
      {norm === 'slide' && <SlideArt fg={accent} />}
      {norm === 'spreadsheet' && <SpreadsheetArt fg={accent} />}
      {norm === 'blog_post' && <BlogPostArt fg={accent} />}
      {norm === 'archive' && <ArchiveArt fg={accent} />}
      {norm === 'audio' && <AudioArt fg={accent} />}
      {norm === 'video' && <VideoArt fg={accent} />}
      {norm === 'default' && <DefaultArt fg={accent} />}
    </svg>
  )
}

export const DeliverableArtwork = memo(DeliverableArtworkImpl)
