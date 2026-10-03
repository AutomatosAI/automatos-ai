'use client'

import { useMemo } from 'react'
import { cn } from '@/lib/utils'
import { AlertTriangle, Coins, Play } from 'lucide-react'
import { Button } from '@/components/ui/button'

interface MissionBudgetBarProps {
  tokensUsed: number
  tokenBudgetEstimate: number
  missionState?: string
  /** Why the mission paused (F153: a budget pause names the spend and the budget). */
  stopDetail?: string | null
  onResume?: () => void
  isResuming?: boolean
  className?: string
}

type BudgetStatus = 'healthy' | 'warning' | 'critical' | 'exceeded'

function getBudgetStatus(percentage: number): BudgetStatus {
  if (percentage > 100) return 'exceeded'
  if (percentage >= 80) return 'critical'
  if (percentage >= 50) return 'warning'
  return 'healthy'
}

const STATUS_STYLES: Record<BudgetStatus, { bar: string; text: string; bg: string }> = {
  healthy: {
    bar: 'bg-success',
    text: 'text-success',
    bg: '',
  },
  warning: {
    bar: 'bg-warning',
    text: 'text-warning',
    bg: '',
  },
  critical: {
    bar: 'bg-destructive',
    text: 'text-destructive',
    bg: '',
  },
  exceeded: {
    bar: 'bg-destructive animate-pulse',
    text: 'text-destructive',
    bg: 'border-destructive/30 bg-destructive/5',
  },
}

/** What the banner says. F153: a paused mission says why. F247: a failed one says
 *  why too, and that Retry runs its failed steps again with the same plan. */
function bannerText(missionState: string | undefined, status: BudgetStatus, stopDetail?: string | null): string {
  if (missionState === 'paused') return stopDetail || 'Mission paused'
  if (missionState === 'failed') {
    return `${(stopDetail || 'Mission failed').replace(/\.$/, '')}. Retry runs its failed steps again, with the same plan.`
  }
  if (status === 'exceeded') return 'Budget exceeded — mission may be paused'
  if (status === 'critical') return 'Budget critical — only synthesis and review tasks will dispatch'
  return 'Budget usage above 50%'
}

export function MissionBudgetBar({
  tokensUsed,
  tokenBudgetEstimate,
  missionState,
  stopDetail,
  onResume,
  isResuming,
  className,
}: MissionBudgetBarProps) {
  const { percentage, status, styles } = useMemo(() => {
    const pct = tokenBudgetEstimate > 0
      ? Math.round((tokensUsed / tokenBudgetEstimate) * 100)
      : 0
    const s = getBudgetStatus(pct)
    return { percentage: pct, status: s, styles: STATUS_STYLES[s] }
  }, [tokensUsed, tokenBudgetEstimate])

  const isPaused = missionState === 'paused'
  // F247: a failed mission is retried from here, through the same Resume.
  const isFailed = missionState === 'failed'
  const stopped = isPaused || isFailed

  return (
    <div className={cn('space-y-2', styles.bg && `rounded-lg border p-3 ${styles.bg}`, className)}>
      {/* Label row */}
      <div className="flex items-center justify-between text-xs">
        <div className="flex items-center gap-1.5">
          <Coins className={cn('w-3.5 h-3.5', styles.text)} />
          <span className="text-muted-foreground">Token Budget</span>
        </div>
        <span className={cn('font-mono font-medium', styles.text)}>
          {tokensUsed.toLocaleString()} / {tokenBudgetEstimate.toLocaleString()} ({percentage}%)
          <span className="ml-2 text-muted-foreground">
            ~${((tokensUsed / 1_000_000) * 4).toFixed(2)}
          </span>
        </span>
      </div>

      {/* Progress bar */}
      <div className="relative h-2 w-full overflow-hidden rounded-full bg-secondary">
        <div
          className={cn('h-full rounded-full transition-all duration-300', styles.bar)}
          style={{ width: `${Math.min(percentage, 100)}%` }}
        />
      </div>

      {/* Warning banner + resume button. F153: a paused mission says why. The
          budget is spend in dollars (a Claude Code session's tokens cost
          nothing), so a budget pause can come at any token percentage. */}
      {(stopped || status === 'warning' || status === 'critical' || status === 'exceeded') && (
        <div className="flex items-center justify-between gap-2">
          <div className={cn('flex items-center gap-1.5 text-[11px]', stopped ? 'text-warning' : styles.text)}>
            <AlertTriangle className="w-3 h-3 shrink-0" />
            <span>{bannerText(missionState, status, stopDetail)}</span>
          </div>
          {stopped && onResume && (
            <Button
              size="sm"
              variant="outline"
              onClick={onResume}
              disabled={isResuming}
              className="h-6 px-2.5 text-[11px] gap-1 border-warning/40 text-warning hover:bg-warning/10"
            >
              <Play className="w-3 h-3" />
              {isFailed ? (isResuming ? 'Retrying...' : 'Retry') : (isResuming ? 'Resuming...' : 'Resume')}
            </Button>
          )}
        </div>
      )}
    </div>
  )
}
