'use client'

import { Code2, Target } from 'lucide-react'
import { cn } from '@/lib/utils'

interface ChatModeBarProps {
  isCodeActive: boolean
  isMissionActive: boolean
  onCodeClick: () => void
  onMissionClick: () => void
}

const pillBase = [
  'inline-flex items-center justify-center rounded-full',
  'min-h-[44px] min-w-[44px] px-3 py-2 md:min-h-0 md:min-w-0 md:px-3 md:py-1.5',
  'gap-2 text-xs font-medium',
  'backdrop-blur transition-colors',
  'shadow-[0_0_18px_rgba(249,115,22,0.10)]',
].join(' ')

const activeStyle = 'bg-orange-500/20 ring-1 ring-orange-500/50 text-foreground/90'
const inactiveStyle = 'bg-black/10 text-foreground/90 hover:bg-orange-500/10'

const iconClass = 'h-4 w-4 md:h-3.5 md:w-3.5 text-orange-400'
const labelClass = 'hidden md:inline'

export function ChatModeBar({
  isCodeActive,
  isMissionActive,
  onCodeClick,
  onMissionClick,
}: ChatModeBarProps) {
  return (
    <div className="flex flex-wrap justify-center gap-3 md:gap-2">
      {/* Code mode */}
      <button
        type="button"
        onClick={onCodeClick}
        title="Code"
        className={cn(pillBase, isCodeActive ? activeStyle : inactiveStyle)}
      >
        <Code2 className={iconClass} />
        <span className={labelClass}>Code</span>
      </button>

      {/* Mission mode */}
      <button
        type="button"
        onClick={onMissionClick}
        title="Mission"
        className={cn(pillBase, isMissionActive ? activeStyle : inactiveStyle)}
      >
        <Target className={iconClass} />
        <span className={labelClass}>Mission</span>
      </button>
    </div>
  )
}
