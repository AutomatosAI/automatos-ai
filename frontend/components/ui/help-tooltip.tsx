'use client'

import React, { useState } from 'react'
import { Info, HelpCircle, ExternalLink } from 'lucide-react'
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'

export interface TooltipHelpProps {
  /** Unique identifier for tooltip content (matches tooltips.json) */
  id: string
  /** Help text to display (if not using id lookup) */
  text?: string
  /** Optional documentation link */
  docLink?: string
  /** Icon variant */
  variant?: 'info' | 'help'
  /** Icon size */
  size?: 'sm' | 'md' | 'lg'
  /** Position of icon relative to text/element */
  position?: 'left' | 'right' | 'top' | 'bottom'
  /** Custom icon component */
  icon?: React.ReactNode
  /** Custom className for trigger button */
  className?: string
  /** Show as inline (no button, just icon) */
  inline?: boolean
}

/**
 * HelpTooltip Component
 * 
 * Displays contextual help with tooltips. Can be used in two modes:
 * 1. With ID: Looks up content from tooltips.json
 * 2. With text: Displays provided text directly
 * 
 * Usage:
 * ```tsx
 * // With ID lookup
 * <HelpTooltip id="agents.roster.create_button" />
 * 
 * // With direct text
 * <HelpTooltip 
 *   text="This is a help message" 
 *   docLink="/docs/user-guides/USER_GUIDE_AGENTS#creating-agents"
 * />
 * 
 * // Inline next to label
 * <label>
 *   Agent Name <HelpTooltip id="agents.create.name" inline />
 * </label>
 * ```
 */
export function HelpTooltip({
  id,
  text,
  docLink,
  variant = 'info',
  size = 'sm',
  position = 'right',
  icon,
  className,
  inline = false,
}: TooltipHelpProps) {
  const [open, setOpen] = useState(false)

  // Icon component based on variant
  const IconComponent = icon || (variant === 'info' ? Info : HelpCircle)

  // Icon size classes
  const iconSizes = {
    sm: 'w-4 h-4',
    md: 'w-5 h-5',
    lg: 'w-6 h-6',
  }

  // Get tooltip content (from useTooltips hook or direct text)
  // Note: In full implementation, would use useTooltips(id) to fetch from tooltips.json
  const tooltipContent = text || `Help content for ${id}`
  const tooltipDocLink = docLink || undefined

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        {inline ? (
          <button
            className={cn(
              'inline-flex items-center justify-center',
              'text-muted-foreground hover:text-foreground',
              'transition-colors cursor-help',
              'ml-1',
              className
            )}
            type="button"
            aria-label="Help information"
          >
            <IconComponent className={iconSizes[size]} />
          </button>
        ) : (
          <Button
            variant="ghost"
            size="icon"
            className={cn(
              'h-8 w-8 rounded-full',
              'text-muted-foreground hover:text-foreground',
              'hover:bg-accent',
              className
            )}
            type="button"
            aria-label="Help information"
          >
            <IconComponent className={iconSizes[size]} />
          </Button>
        )}
      </PopoverTrigger>
      <PopoverContent
        side={position}
        className={cn(
          'w-80 p-4',
          'glass-card',
          'border border-border/50'
        )}
        sideOffset={5}
      >
        <div className="space-y-3">
          {/* Help Title (if available) */}
          {id && (
            <div className="flex items-center space-x-2">
              <IconComponent className="w-5 h-5 text-primary" />
              <h4 className="font-semibold text-sm">Help</h4>
            </div>
          )}

          {/* Help Content */}
          <div className="text-sm text-muted-foreground leading-relaxed">
            {tooltipContent}
          </div>

          {/* Documentation Link */}
          {tooltipDocLink && (
            <div className="pt-2 border-t border-border/30">
              <a
                href={tooltipDocLink}
                className={cn(
                  'inline-flex items-center space-x-1',
                  'text-xs text-primary hover:text-primary/80',
                  'transition-colors'
                )}
                target="_blank"
                rel="noopener noreferrer"
                onClick={() => setOpen(false)}
              >
                <span>View full documentation</span>
                <ExternalLink className="w-3 h-3" />
              </a>
            </div>
          )}
        </div>
      </PopoverContent>
    </Popover>
  )
}

/**
 * Inline variant for use in forms and labels
 */
export function InlineHelp(props: Omit<TooltipHelpProps, 'inline'>) {
  return <HelpTooltip {...props} inline />
}

/**
 * Section help - larger, for section headers
 */
export function SectionHelp(props: Omit<TooltipHelpProps, 'size'>) {
  return <HelpTooltip {...props} size="md" />
}

/**
 * Field help - for form fields
 */
export function FieldHelp(props: Omit<TooltipHelpProps, 'variant' | 'inline' | 'size'>) {
  return <HelpTooltip {...props} variant="help" inline size="sm" />
}

