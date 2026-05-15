'use client'

/**
 * Sparkline — tiny inline trend line used in the StatsStrip and elsewhere.
 *
 * Pure SVG, no external dependencies. Accepts a [number] series and renders
 * a smoothed polyline scaled to the available box. Tone drives stroke color
 * via the studio semantic palette (ok=olive, err=accent, info=navy, etc).
 */

export type SparklineTone = 'ok' | 'err' | 'warn' | 'info' | 'muted'

interface SparklineProps {
  data: number[]
  tone?: SparklineTone
  className?: string
  height?: number
}

const TONE_TO_COLOR: Record<SparklineTone, string> = {
  ok: 'hsl(82 50% 22%)',
  err: 'hsl(var(--accent))',
  warn: 'hsl(38 78% 27%)',
  info: 'hsl(var(--info))',
  muted: 'hsl(var(--muted-foreground))',
}

export function Sparkline({ data, tone = 'info', className, height = 18 }: SparklineProps) {
  if (!data || data.length < 2) {
    return null
  }
  const w = 80
  const h = height
  const max = Math.max(...data, 1)
  const min = Math.min(...data, 0)
  const range = max - min || 1
  const step = w / (data.length - 1)
  const pts = data.map<[number, number]>((v, i) => [
    i * step,
    h - ((v - min) / range) * h,
  ])
  const path = pts
    .map((p, i) => (i === 0 ? `M${p[0]},${p[1]}` : `L${p[0]},${p[1]}`))
    .join(' ')
  const stroke = TONE_TO_COLOR[tone]
  const last = pts[pts.length - 1]

  return (
    <svg
      viewBox={`0 0 ${w} ${h}`}
      preserveAspectRatio="none"
      className={className}
      style={{ width: '100%', height: '100%' }}
    >
      <path
        d={path}
        fill="none"
        stroke={stroke}
        strokeWidth="1.4"
        strokeLinejoin="round"
        strokeLinecap="round"
      />
      <circle cx={last[0]} cy={last[1]} r="1.6" fill={stroke} />
    </svg>
  )
}
