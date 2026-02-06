import { cn } from '@/lib/utils'

const statusStyles = {
  success: 'bg-[hsl(var(--success))]/10 text-[hsl(var(--success))] border-[hsl(var(--success))]/20',
  active: 'bg-[hsl(var(--info))]/10 text-[hsl(var(--info))] border-[hsl(var(--info))]/20',
  warning: 'bg-[hsl(var(--warning))]/10 text-[hsl(var(--warning))] border-[hsl(var(--warning))]/20',
  error: 'bg-[hsl(var(--destructive))]/10 text-[hsl(var(--destructive))] border-[hsl(var(--destructive))]/20',
  info: 'bg-[hsl(var(--info))]/10 text-[hsl(var(--info))] border-[hsl(var(--info))]/20',
  neutral: 'bg-secondary/50 text-muted-foreground border-border/30',
  purple: 'bg-[hsl(var(--agent))]/10 text-[hsl(var(--agent))] border-[hsl(var(--agent))]/20',
  primary: 'bg-primary/10 text-primary border-primary/20',
} as const

export type StatusVariant = keyof typeof statusStyles

export interface StatusBadgeProps {
  status: StatusVariant
  children: React.ReactNode
  dot?: boolean
  size?: 'sm' | 'default'
  className?: string
}

export function StatusBadge({
  status,
  children,
  dot = false,
  size = 'default',
  className,
}: StatusBadgeProps) {
  return (
    <span
      className={cn(
        'inline-flex items-center rounded-full border font-semibold transition-colors',
        size === 'sm' ? 'px-2 py-0.5 text-[10px]' : 'px-2.5 py-0.5 text-xs',
        statusStyles[status],
        className
      )}
    >
      {dot && (
        <span
          className={cn(
            'rounded-full mr-1.5 animate-pulse',
            size === 'sm' ? 'w-1.5 h-1.5' : 'w-2 h-2',
            // dot color inherits from text via currentColor
            'bg-current'
          )}
        />
      )}
      {children}
    </span>
  )
}
