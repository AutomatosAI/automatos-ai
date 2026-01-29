/**
 * SuggestionChip Component (PRD-40: Dynamic Tool Suggestions)
 *
 * Individual suggestion chip that displays a single prompt suggestion.
 * Used within ToolSuggestionBar to show clickable suggestion prompts.
 */

import * as React from 'react'
import { cn } from '@/lib/utils'

export interface SuggestionChipProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  text: string
}

export function SuggestionChip({ text, className, onClick, ...props }: SuggestionChipProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        // Base styles - match bottom action buttons
        'inline-flex items-center justify-center',
        'px-4 py-2',
        'text-sm font-normal',
        'rounded-full',
        'border border-orange-500/20',
        'bg-background/80',

        // Consistent sizing
        'min-w-fit',
        'shrink-0',

        // Hover effects
        'hover:border-orange-500/40',
        'hover:bg-orange-500/5',

        // Active/pressed state
        'active:scale-95',

        // Transitions
        'transition-all duration-200',

        // Focus styles
        'focus-visible:outline-none',
        'focus-visible:ring-2',
        'focus-visible:ring-orange-500/50',

        // Disabled state
        'disabled:pointer-events-none',
        'disabled:opacity-50',

        // Text - keep on one line
        'whitespace-nowrap',
        'cursor-pointer',

        className
      )}
      {...props}
    >
      {text}
    </button>
  )
}
