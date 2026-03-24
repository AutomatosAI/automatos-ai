'use client'

import { useMemo } from 'react'
import { cn } from '@/lib/utils'
import { AlertTriangle, Coins, Play } from 'lucide-react'
import { Button } from '@/components/ui/button'

interface MissionBudgetBarProps {
  tokensUsed: number
  tokenBudgetEstimate: number
  missionState?: string
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
    bar: 'bg-green-500',
    text: 'text-green-400',
    bg: '',
  },
  warning: {
    bar: 'bg-amber-500',
    text: 'text-amber-400',
    bg: '',
  },
  critical: {
    bar: 'bg-red-500',
    text: 'text-red-400',
    bg: '',
  },
  exceeded: {
    bar: 'bg-red-500 animate-pulse',
    text: 'text-red-400',
    bg: 'border-red-500/30 bg-red-500/5',
  },
}

export function MissionBudgetBar({
  tokensUsed,
  tokenBudgetEstimate,
  missionState,
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

  return (
    <div className={cn('space-y-1.5', styles.bg && `rounded-lg border p-3 ${styles.bg}`, className)}>
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
          className={cn('h-full rounded-full transition-all duration-500', styles.bar)}
          style={{ width: `${Math.min(percentage, 100)}%` }}
        />
      </div>

      {/* Warning banner + resume button */}
      {(status === 'warning' || status === 'critical' || status === 'exceeded') && (
        <div className="flex items-center justify-between gap-2">
          <div className={cn('flex items-center gap-1.5 text-[11px]', styles.text)}>
            <AlertTriangle className="w-3 h-3 shrink-0" />
            <span>
              {isPaused
                ? 'Mission paused — budget exceeded'
                : status === 'exceeded'
                  ? 'Budget exceeded — mission may be paused'
                  : status === 'critical'
                    ? 'Budget critical — only synthesis and review tasks will dispatch'
                    : 'Budget usage above 50%'}
            </span>
          </div>
          {isPaused && onResume && (
            <Button
              size="sm"
              variant="outline"
              onClick={onResume}
              disabled={isResuming}
              className="h-6 px-2.5 text-[11px] gap-1 border-amber-500/40 text-amber-400 hover:bg-amber-500/10"
            >
              <Play className="w-3 h-3" />
              {isResuming ? 'Resuming...' : 'Resume'}
            </Button>
          )}
        </div>
      )}
    </div>
  )
}
