'use client'

/**
 * DeliverableArtwork
 * ==================
 *
 * Static SVG artwork for deliverable cards. Each artifact_type gets
 * a unique illustrated tile with type-specific glyph + accent colour.
 *
 * - 100% width/height SVG, square aspect (parent enforces aspect-square)
 * - Dark canvas + type-specific accent colour
 * - Static — no animations (a wall of these tiles needs to feel calm)
 * - Deterministic IDs via useId() so multiple cards on a page don't collide
 */

import { memo, useId } from 'react'

type ArtworkType =
  | 'report'
  | 'document'
  | 'image'
  | 'code'
  | 'slide'
  | 'spreadsheet'
  | 'archive'
  | 'audio'
  | 'video'
  | 'default'

interface DeliverableArtworkProps {
  type: string
  className?: string
}

const TYPE_ACCENT: Record<ArtworkType, { fg: string; bg: string }> = {
  report: { fg: '#60a5fa', bg: '#0c1626' },
  document: { fg: '#cbd5e1', bg: '#16181c' },
  image: { fg: '#c084fc', bg: '#1a1326' },
  code: { fg: '#34d399', bg: '#0c1f1a' },
  slide: { fg: '#fb923c', bg: '#1f1308' },
  spreadsheet: { fg: '#4ade80', bg: '#0c1f12' },
  archive: { fg: '#fbbf24', bg: '#1f1808' },
  audio: { fg: '#f472b6', bg: '#1f0c1a' },
  video: { fg: '#f87171', bg: '#1f0c0c' },
  default: { fg: '#94a3b8', bg: '#16181c' },
}

function normalizeType(t: string): ArtworkType {
  if (t in TYPE_ACCENT) return t as ArtworkType
  return 'default'
}

// ─── Per-type artwork pieces ────────────────────────────────────────

function ReportArt({ fg }: { fg: string }) {
  const bars = [
    { x: 50, h: 40 },
    { x: 80, h: 70 },
    { x: 110, h: 55 },
    { x: 140, h: 90 },
  ]
  return (
    <g>
      <line x1="40" y1="150" x2="160" y2="150" stroke={fg} strokeOpacity="0.3" strokeWidth="1" />
      {bars.map((b, i) => (
        <rect
          key={i}
          x={b.x}
          y={150 - b.h}
          width="18"
          height={b.h}
          rx="2"
          fill={fg}
          fillOpacity="0.7"
        />
      ))}
      <polyline
        points="50,70 80,50 110,60 140,30"
        fill="none"
        stroke={fg}
        strokeWidth="1.5"
        strokeOpacity="0.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </g>
  )
}

function DocumentArt({ fg }: { fg: string }) {
  return (
    <g>
      <path
        d="M 70 40 L 130 40 L 145 55 L 145 160 L 70 160 Z"
        fill={fg}
        fillOpacity="0.08"
        stroke={fg}
        strokeOpacity="0.5"
        strokeWidth="1.5"
      />
      <path d="M 130 40 L 130 55 L 145 55" fill="none" stroke={fg} strokeOpacity="0.5" strokeWidth="1.5" />
      {[80, 100, 120, 140].map((y, i) => (
        <line
          key={y}
          x1="80"
          y1={y}
          x2={i === 0 ? '120' : '135'}
          y2={y}
          stroke={fg}
          strokeOpacity="0.4"
          strokeWidth="1.5"
          strokeLinecap="round"
        />
      ))}
    </g>
  )
}

function ImageArt({ fg }: { fg: string }) {
  return (
    <g>
      <rect x="50" y="50" width="100" height="100" rx="6" fill={fg} fillOpacity="0.08" stroke={fg} strokeOpacity="0.5" strokeWidth="1.5" />
      <circle cx="125" cy="80" r="10" fill={fg} fillOpacity="0.7" />
      <path
        d="M 50 140 L 80 105 L 105 125 L 130 95 L 150 125 L 150 150 L 50 150 Z"
        fill={fg}
        fillOpacity="0.4"
      />
    </g>
  )
}

function CodeArt({ fg }: { fg: string }) {
  return (
    <g>
      <text
        x="60"
        y="120"
        fontFamily="ui-monospace, monospace"
        fontSize="60"
        fontWeight="500"
        fill={fg}
        fillOpacity="0.7"
      >
        {'{'}
      </text>
      <text
        x="120"
        y="120"
        fontFamily="ui-monospace, monospace"
        fontSize="60"
        fontWeight="500"
        fill={fg}
        fillOpacity="0.7"
      >
        {'}'}
      </text>
      <line x1="55" y1="145" x2="80" y2="145" stroke={fg} strokeOpacity="0.4" strokeWidth="1.5" />
      <line x1="85" y1="145" x2="120" y2="145" stroke={fg} strokeOpacity="0.25" strokeWidth="1.5" />
      <line x1="125" y1="145" x2="145" y2="145" stroke={fg} strokeOpacity="0.4" strokeWidth="1.5" />
    </g>
  )
}

function SlideArt({ fg }: { fg: string }) {
  return (
    <g>
      <rect x="55" y="55" width="90" height="65" rx="4" fill={fg} fillOpacity="0.12" stroke={fg} strokeOpacity="0.3" strokeWidth="1" />
      <rect x="60" y="65" width="90" height="65" rx="4" fill={fg} fillOpacity="0.18" stroke={fg} strokeOpacity="0.4" strokeWidth="1" />
      <rect x="65" y="75" width="90" height="65" rx="4" fill={fg} fillOpacity="0.25" stroke={fg} strokeOpacity="0.6" strokeWidth="1.5" />
      <line x1="75" y1="92" x2="135" y2="92" stroke={fg} strokeOpacity="0.7" strokeWidth="2" strokeLinecap="round" />
      <line x1="75" y1="108" x2="120" y2="108" stroke={fg} strokeOpacity="0.5" strokeWidth="1.5" />
      <line x1="75" y1="120" x2="125" y2="120" stroke={fg} strokeOpacity="0.5" strokeWidth="1.5" />
    </g>
  )
}

function SpreadsheetArt({ fg }: { fg: string }) {
  const cellW = 25
  const cellH = 22
  const cols = 4
  const rows = 4
  const startX = 52
  const startY = 50
  return (
    <g>
      {Array.from({ length: rows }).map((_, r) =>
        Array.from({ length: cols }).map((__, c) => {
          const isHighlight = r === 1 && c === 2
          return (
            <rect
              key={`${r}-${c}`}
              x={startX + c * cellW}
              y={startY + r * cellH}
              width={cellW}
              height={cellH}
              fill={fg}
              fillOpacity={isHighlight ? 0.4 : r === 0 ? 0.18 : 0.08}
              stroke={fg}
              strokeOpacity="0.4"
              strokeWidth="0.8"
            />
          )
        }),
      )}
    </g>
  )
}

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
      {bars.map((b, i) => (
        <rect
          key={i}
          x={b.x}
          y={100 - b.base / 2}
          width="8"
          height={b.base}
          rx="2"
          fill={fg}
          fillOpacity="0.7"
        />
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
      <path
        d="M 75 50 L 115 50 L 135 70 L 135 150 L 75 150 Z"
        fill={fg}
        fillOpacity="0.1"
        stroke={fg}
        strokeOpacity="0.5"
        strokeWidth="1.5"
      />
      <path d="M 115 50 L 115 70 L 135 70" fill="none" stroke={fg} strokeOpacity="0.5" strokeWidth="1.5" />
    </g>
  )
}

// ─── Main component ─────────────────────────────────────────────────

function DeliverableArtworkImpl({ type, className }: DeliverableArtworkProps) {
  const id = useId().replace(/:/g, '')
  const norm = normalizeType(type)
  const accent = TYPE_ACCENT[norm]
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
        <radialGradient id={gradId} cx="50%" cy="50%" r="60%">
          <stop offset="0%" stopColor={accent.fg} stopOpacity="0.18" />
          <stop offset="100%" stopColor={accent.bg} stopOpacity="0" />
        </radialGradient>
      </defs>

      <rect x="0" y="0" width="200" height="200" fill={accent.bg} />
      <rect x="0" y="0" width="200" height="200" fill={`url(#${gradId})`} />

      {norm === 'report' && <ReportArt fg={accent.fg} />}
      {norm === 'document' && <DocumentArt fg={accent.fg} />}
      {norm === 'image' && <ImageArt fg={accent.fg} />}
      {norm === 'code' && <CodeArt fg={accent.fg} />}
      {norm === 'slide' && <SlideArt fg={accent.fg} />}
      {norm === 'spreadsheet' && <SpreadsheetArt fg={accent.fg} />}
      {norm === 'archive' && <ArchiveArt fg={accent.fg} />}
      {norm === 'audio' && <AudioArt fg={accent.fg} />}
      {norm === 'video' && <VideoArt fg={accent.fg} />}
      {norm === 'default' && <DefaultArt fg={accent.fg} />}
    </svg>
  )
}

export const DeliverableArtwork = memo(DeliverableArtworkImpl)
