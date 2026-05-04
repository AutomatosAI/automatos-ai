'use client'

/**
 * Marketplace — Featured Showcase Card
 *
 * Hero card for the Marketplace Featured slot. Theatrical "look at this"
 * pedestal visual: spotlight cone, rim light, drifting dust motes,
 * a glassy hero tile in the center. Item copy + Install CTA overlay.
 *
 * Mirrors the spirit of MissionConstellationCard on /assignments — same
 * dark canvas, orange brand accents, animated SVG, no JS timers.
 */

import { useId } from 'react'
import { motion } from 'framer-motion'
import { Star, Download } from 'lucide-react'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import type { MarketplaceItem } from './marketplace-homepage'

interface FeaturedShowcaseCardProps {
  item: MarketplaceItem
  isAdmin: boolean
  onItemClick: (id: number) => void
  onToggleFeatured: () => void
  toggleDisabled?: boolean
}

// Deterministic dust motes (no Math.random — SSR safe)
const MOTES = Array.from({ length: 28 }, (_, i) => {
  const seed = (i * 9301 + 49297) % 233280
  const r1 = seed / 233280
  const r2 = ((seed * 9301 + 49297) % 233280) / 233280
  const r3 = ((seed * 13 + 17) % 233280) / 233280
  const r4 = ((seed * 7 + 5) % 233280) / 233280
  return {
    x: 120 + r1 * 480,
    y: 220 + r2 * 100,
    r: 0.6 + r3 * 1.4,
    dur: 5 + r4 * 6,
    delay: r1 * 5,
    drift: 4 + r3 * 8,
  }
})

function formatInstalls(n: number): string {
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`
  return n.toString()
}

export function FeaturedShowcaseCard({
  item,
  isAdmin,
  onItemClick,
  onToggleFeatured,
  toggleDisabled = false,
}: FeaturedShowcaseCardProps) {
  // Unique gradient/filter IDs so the SVG can mount alongside others
  const uid = useId().replace(/:/g, '')
  const stageId = `ms-stage-${uid}`
  const spotId = `ms-spot-${uid}`
  const poolId = `ms-pool-${uid}`
  const tileId = `ms-tile-${uid}`
  const tileEdgeId = `ms-tile-edge-${uid}`
  const glowId = `ms-glow-${uid}`
  const softId = `ms-soft-${uid}`

  const handleInstall = (e: React.MouseEvent) => {
    e.stopPropagation()
    onItemClick(item.id)
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.1 }}
      className="lg:col-span-2"
    >
      <Card
        role="button"
        tabIndex={0}
        onClick={() => onItemClick(item.id)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault()
            onItemClick(item.id)
          }
        }}
        className="relative overflow-hidden cursor-pointer border-primary/20 bg-gradient-to-br from-[#14110f] via-[#0e0d12] to-[#0a0a0e] min-h-[260px] hover:border-primary/40 transition-colors group focus:outline-none focus:ring-2 focus:ring-primary/40 card-glow"
      >
        {/* ── Stage visual ────────────────────────────────────── */}
        <svg
          viewBox="0 0 720 300"
          preserveAspectRatio="xMidYMid slice"
          className="absolute inset-0 w-full h-full"
          aria-hidden="true"
        >
          <defs>
            <linearGradient id={stageId} x1="0" x2="0" y1="0" y2="1">
              <stop offset="0%" stopColor="#1a120c" stopOpacity="0" />
              <stop offset="60%" stopColor="#1a120c" stopOpacity="0.4" />
              <stop offset="100%" stopColor="#0a0a0e" stopOpacity="0.95" />
            </linearGradient>
            <linearGradient id={spotId} x1="0.5" x2="0.5" y1="0" y2="1">
              <stop offset="0%" stopColor="#fed7aa" stopOpacity="0.55" />
              <stop offset="60%" stopColor="#fdba74" stopOpacity="0.18" />
              <stop offset="100%" stopColor="#f97316" stopOpacity="0" />
            </linearGradient>
            <radialGradient id={poolId} cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor="#fed7aa" stopOpacity="0.35" />
              <stop offset="60%" stopColor="#fb923c" stopOpacity="0.12" />
              <stop offset="100%" stopColor="#7c2d12" stopOpacity="0" />
            </radialGradient>
            <linearGradient id={tileId} x1="0" x2="1" y1="0" y2="1">
              <stop offset="0%" stopColor="#26201a" />
              <stop offset="100%" stopColor="#14110f" />
            </linearGradient>
            <linearGradient id={tileEdgeId} x1="0" x2="0" y1="0" y2="1">
              <stop offset="0%" stopColor="#fdba74" stopOpacity="0.55" />
              <stop offset="100%" stopColor="#f97316" stopOpacity="0.08" />
            </linearGradient>
            <filter id={glowId} x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="4" />
            </filter>
            <filter id={softId} x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="1.5" />
            </filter>
          </defs>

          {/* stage backdrop */}
          <rect width="720" height="300" fill={`url(#${stageId})`} />

          {/* spotlight cone */}
          <polygon
            points="320,0 400,0 540,300 180,300"
            fill={`url(#${spotId})`}
            opacity="0.9"
          >
            <animate
              attributeName="opacity"
              values="0.7;1;0.7"
              dur="6s"
              repeatCount="indefinite"
            />
          </polygon>

          {/* floor pool of light */}
          <ellipse cx="360" cy="248" rx="180" ry="22" fill={`url(#${poolId})`}>
            <animate
              attributeName="rx"
              values="170;195;170"
              dur="6s"
              repeatCount="indefinite"
            />
            <animate
              attributeName="opacity"
              values="0.7;1;0.7"
              dur="6s"
              repeatCount="indefinite"
            />
          </ellipse>

          {/* horizon line */}
          <line
            x1="0"
            y1="248"
            x2="720"
            y2="248"
            stroke="rgba(249,115,22,0.18)"
            strokeWidth="1"
          />
          <line
            x1="0"
            y1="249.5"
            x2="720"
            y2="249.5"
            stroke="rgba(255,255,255,0.04)"
            strokeWidth="1"
          />

          {/* drifting dust motes */}
          <g fill="#fed7aa">
            {MOTES.map((m, i) => (
              <circle key={i} cx={m.x} cy={m.y} r={m.r} opacity="0.5">
                <animate
                  attributeName="cy"
                  values={`${m.y};${m.y - 80 - m.drift * 4};${m.y}`}
                  dur={`${m.dur}s`}
                  begin={`${m.delay}s`}
                  repeatCount="indefinite"
                />
                <animate
                  attributeName="opacity"
                  values="0;0.7;0"
                  dur={`${m.dur}s`}
                  begin={`${m.delay}s`}
                  repeatCount="indefinite"
                />
              </circle>
            ))}
          </g>

          {/* rim light behind hero tile */}
          <g transform="translate(360 168)">
            <circle r="78" fill="#f97316" opacity="0.18" filter={`url(#${glowId})`}>
              <animate
                attributeName="opacity"
                values="0.12;0.28;0.12"
                dur="5s"
                repeatCount="indefinite"
              />
            </circle>
            <circle r="56" fill="#fb923c" opacity="0.15" filter={`url(#${glowId})`}>
              <animate
                attributeName="opacity"
                values="0.1;0.22;0.1"
                dur="5s"
                begin="0.6s"
                repeatCount="indefinite"
              />
            </circle>
          </g>

          {/* hero tile */}
          <g transform="translate(360 168)">
            {/* shadow */}
            <ellipse
              cx="0"
              cy="78"
              rx="52"
              ry="6"
              fill="#000"
              opacity="0.55"
              filter={`url(#${softId})`}
            />
            {/* tile body */}
            <rect
              x="-56"
              y="-66"
              width="112"
              height="132"
              rx="14"
              fill={`url(#${tileId})`}
              stroke="rgba(253,186,116,0.35)"
              strokeWidth="1"
            />
            {/* top edge highlight */}
            <rect x="-56" y="-66" width="112" height="2" fill={`url(#${tileEdgeId})`} />

            {/* glyph: emoji icon if present, else stylized stacked sheets */}
            {item.icon ? (
              <text
                x="0"
                y="14"
                textAnchor="middle"
                fontSize="64"
                style={{
                  fontFamily:
                    '"Apple Color Emoji", "Segoe UI Emoji", "Noto Color Emoji", system-ui, sans-serif',
                }}
              >
                {item.icon}
              </text>
            ) : (
              <g
                stroke="#fdba74"
                strokeWidth="2"
                fill="none"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <rect x="-26" y="-30" width="48" height="36" rx="3" opacity="0.35" />
                <rect x="-22" y="-22" width="48" height="36" rx="3" opacity="0.6" />
                <rect x="-18" y="-14" width="48" height="36" rx="3" opacity="1">
                  <animate
                    attributeName="opacity"
                    values="0.7;1;0.7"
                    dur="4s"
                    repeatCount="indefinite"
                  />
                </rect>
                <line
                  x1="-10"
                  y1="-2"
                  x2="22"
                  y2="-2"
                  stroke="#fed7aa"
                  strokeWidth="1.4"
                  opacity="0.7"
                />
                <line
                  x1="-10"
                  y1="6"
                  x2="14"
                  y2="6"
                  stroke="#fed7aa"
                  strokeWidth="1.4"
                  opacity="0.5"
                />
                <line
                  x1="-10"
                  y1="14"
                  x2="18"
                  y2="14"
                  stroke="#fed7aa"
                  strokeWidth="1.4"
                  opacity="0.6"
                />
              </g>
            )}
            {/* corner specular */}
            <circle cx="-40" cy="-50" r="4" fill="#fff7ed" opacity="0.55" />
          </g>

          {/* drifting horizontal volumetric streaks */}
          <g>
            <rect x="0" y="120" width="80" height="0.8" fill="#fdba74" opacity="0" rx="1">
              <animate attributeName="x" values="-100;820" dur="9s" repeatCount="indefinite" />
              <animate
                attributeName="opacity"
                values="0;0.45;0"
                dur="9s"
                repeatCount="indefinite"
              />
            </rect>
            <rect x="0" y="195" width="60" height="0.7" fill="#fed7aa" opacity="0" rx="1">
              <animate
                attributeName="x"
                values="-100;820"
                dur="11s"
                begin="3s"
                repeatCount="indefinite"
              />
              <animate
                attributeName="opacity"
                values="0;0.35;0"
                dur="11s"
                begin="3s"
                repeatCount="indefinite"
              />
            </rect>
          </g>
        </svg>

        {/* ── Bottom gradient mask for copy legibility ─────────── */}
        <div
          className="absolute inset-x-0 bottom-0 h-3/5 pointer-events-none"
          style={{
            background:
              'linear-gradient(180deg, rgba(14,13,18,0) 0%, rgba(14,13,18,0.78) 65%, rgba(14,13,18,0.95) 100%)',
          }}
        />

        {/* ── Admin toggle (top-right) ──────────────────────────── */}
        {isAdmin && (
          <Button
            variant="ghost"
            size="sm"
            className="absolute top-3 right-3 z-10 text-primary hover:text-primary/80 hover:bg-primary/10"
            onClick={(e) => {
              e.stopPropagation()
              onToggleFeatured()
            }}
            disabled={toggleDisabled}
          >
            <Star className="h-4 w-4 fill-current" />
          </Button>
        )}

        {/* ── Copy overlay (bottom-left) ────────────────────────── */}
        <div className="absolute left-6 bottom-5 right-44 z-10 flex flex-col gap-2.5">
          <div className="flex items-center gap-2.5">
            <div className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded font-mono text-[10px] tracking-[0.12em] text-orange-300 border border-primary/30 bg-primary/10">
              <span className="w-1 h-1 rounded-full bg-primary shadow-[0_0_6px_rgba(249,115,22,0.9)]" />
              FEATURED
            </div>
            {(item.category || item.type) && (
              <span className="font-mono text-[10px] tracking-wide text-slate-300 px-2.5 py-0.5 rounded-full border border-white/10 bg-[rgba(20,17,15,0.6)] backdrop-blur-sm">
                {item.category || item.type}
              </span>
            )}
          </div>
          <h3 className="text-2xl md:text-3xl font-semibold tracking-tight text-foreground leading-tight line-clamp-1">
            {item.name}
          </h3>
          <p className="text-sm text-muted-foreground leading-relaxed line-clamp-2 max-w-xl">
            {item.description}
          </p>
          <div className="flex items-center gap-2 font-mono text-[10.5px] text-slate-400 mt-0.5">
            {item.creator_name && (
              <>
                <span className="text-slate-300">{item.creator_name}</span>
                <span className="text-slate-600">·</span>
              </>
            )}
            <span>v{item.version}</span>
            <span className="text-slate-600">·</span>
            <span className="flex items-center gap-1">
              <Download className="h-3 w-3" />
              {formatInstalls(item.install_count)} installs
            </span>
          </div>
        </div>

        {/* ── Install CTA (bottom-right) ────────────────────────── */}
        <div className="absolute right-6 bottom-5 z-10">
          <Button
            onClick={handleInstall}
            className="shadow-[0_8px_20px_rgba(249,115,22,0.35)]"
          >
            Install
            <svg
              width="14"
              height="14"
              viewBox="0 0 14 14"
              fill="none"
              className="ml-1"
              aria-hidden="true"
            >
              <path
                d="M7 2V10M7 10L3 6M7 10L11 6M2 12H12"
                stroke="currentColor"
                strokeWidth="1.6"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </Button>
        </div>
      </Card>
    </motion.div>
  )
}
