'use client'

/**
 * PRD-244 D5 — the small-width face of "Auto now": the two counts that need a
 * human (open questions, decisions) on the control that opens the rail.
 */
import { PanelRightClose, PanelRightOpen } from 'lucide-react'
import { useAutoNow } from '@/hooks/use-auto-now'
import { cn } from '@/lib/utils'

interface AutoNowPillProps {
  open: boolean
  onToggle: () => void
  className?: string
  style?: React.CSSProperties
}

export function AutoNowPill({ open, onToggle, className, style }: AutoNowPillProps) {
  const { questionCount, decisionsTotal } = useAutoNow()
  const needsYou = questionCount + decisionsTotal
  const Icon = open ? PanelRightClose : PanelRightOpen
  return (
    <button
      type="button"
      onClick={onToggle}
      aria-pressed={open}
      aria-label={open ? 'Hide Auto now rail' : 'Show Auto now rail'}
      title={`Auto now · ${questionCount} question${questionCount === 1 ? '' : 's'} · ${decisionsTotal} decision${decisionsTotal === 1 ? '' : 's'}`}
      className={cn('inline-flex items-center gap-1.5', className)}
      style={style}
    >
      <Icon style={{ width: 14, height: 14, strokeWidth: 1.6 }} />
      <span>Auto now</span>
      {needsYou > 0 && <span className="auto-now-count">{needsYou}</span>}
    </button>
  )
}
