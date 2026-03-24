'use client'

import { useMemo } from 'react'
import { cn } from '@/lib/utils'
import { AlertTriangle, Coins } from 'lucide-react'

interface MissionBudgetBarProps {
  tokensUsed: number
  tokenBudgetEstimate: number
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
  className,
}: MissionBudgetBarProps) {
  const { percentage, status, styles } = useMemo(() => {
    const pct = tokenBudgetEstimate > 0
      ? Math.round((tokensUsed / tokenBudgetEstimate) * 100)
      : 0
    const s = getBudgetStatus(pct)
    return { percentage: pct, status: s, styles: STATUS_STYLES[s] }
  }, [tokensUsed, tokenBudgetEstimate])

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
        </span>
      </div>

      {/* Progress bar */}
      <div className="relative h-2 w-full overflow-hidden rounded-full bg-secondary">
        <div
          className={cn('h-full rounded-full transition-all duration-500', styles.bar)}
          style={{ width: `${Math.min(percentage, 100)}%` }}
        />
      </div>

      {/* Warning banner */}
      {(status === 'warning' || status === 'critical' || status === 'exceeded') && (
        <div className={cn('flex items-center gap-1.5 text-[11px]', styles.text)}>
          <AlertTriangle className="w-3 h-3 shrink-0" />
          <span>
            {status === 'exceeded'
              ? 'Budget exceeded — mission may be paused'
              : status === 'critical'
                ? 'Budget critical — only synthesis and review tasks will dispatch'
                : 'Budget usage above 50%'}
          </span>
        </div>
      )}
    </div>
  )
}
