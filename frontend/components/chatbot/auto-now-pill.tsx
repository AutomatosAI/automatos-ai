'use client'

/**
 * PRD-244 D5 — the small-width face of "Auto now": what needs a human, on the
 * control that opens the rail. PRD-252 R5: that is the one Needs-you number,
 * the one the Board tab, ATTENTION and the Needs you widget show.
 */
import { PanelRightClose, PanelRightOpen } from 'lucide-react'
import { useAutoNow } from '@/hooks/use-auto-now'
import { needsYouBreakdown } from '@/lib/needs-you-breakdown'
import { cn } from '@/lib/utils'

interface AutoNowPillProps {
  open: boolean
  onToggle: () => void
  className?: string
  style?: React.CSSProperties
}

export function AutoNowPill({ open, onToggle, className, style }: AutoNowPillProps) {
  const { needsYou, needsYouTotal } = useAutoNow()
  const Icon = open ? PanelRightClose : PanelRightOpen
  return (
    <button
      type="button"
      onClick={onToggle}
      aria-pressed={open}
      aria-label={open ? 'Hide Auto now rail' : 'Show Auto now rail'}
      title={needsYouTotal > 0 ? `Auto now · needs you: ${needsYouBreakdown(needsYou?.counts)}` : 'Auto now · nothing needs you'}
      className={cn('inline-flex items-center gap-1.5', className)}
      style={style}
    >
      <Icon style={{ width: 14, height: 14, strokeWidth: 1.6 }} />
      <span>Auto now</span>
      {needsYouTotal > 0 && <span className="auto-now-count">{needsYouTotal}</span>}
    </button>
  )
}
