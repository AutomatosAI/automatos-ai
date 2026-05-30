'use client'

// PRD-142 Wave 0 (US-007) — the "Is it working?" answer strip.
// Five tiles over the real measurement endpoints shipped in Wave 0. The
// fifth tile (per-primitive health, US-006) has no honest data source yet
// and is deferred to Wave 3, so it renders as an explicit "not yet
// measured" placeholder rather than a fake green.

import { motion } from 'framer-motion'
import {
  Zap,
  CheckCircle,
  AlertTriangle,
  MessageSquare,
  Boxes,
} from 'lucide-react'
import { Card, CardContent } from '../ui/card'
import { Badge } from '../ui/badge'
import {
  useActivationMetrics,
  useMissionSuccessRate,
  useErrorsBySubsystem,
  useWidgetEngagement,
} from '../../hooks/use-analytics-api'

interface TileProps {
  icon: React.ComponentType<{ className?: string }>
  label: string
  value: React.ReactNode
  sub: React.ReactNode
  accent: string
  isLoading?: boolean
  isError?: boolean
  muted?: boolean
  delay?: number
}

function Tile({ icon: Icon, label, value, sub, accent, isLoading, isError, muted, delay = 0 }: TileProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay }}
    >
      <Card className={`glass-card h-full ${muted ? 'opacity-70' : 'hover:border-primary/20 transition-all duration-300'}`}>
        <CardContent className="p-4">
          <div className="flex items-center justify-between gap-3">
            <div className="flex items-center gap-3 min-w-0">
              <div className="w-10 h-10 rounded-2xl bg-foreground/5 border border-border/60 flex items-center justify-center shrink-0">
                <Icon className={`w-5 h-5 ${accent}`} />
              </div>
              <div className="min-w-0">
                {isLoading ? (
                  <div className="space-y-2">
                    <div className="h-6 w-16 bg-muted rounded animate-pulse" />
                    <div className="h-3 w-24 bg-muted rounded animate-pulse" />
                  </div>
                ) : (
                  <>
                    <div className="text-2xl font-bold leading-none">{isError ? '—' : value}</div>
                    <div className="text-sm text-muted-foreground truncate">{label}</div>
                  </>
                )}
              </div>
            </div>
            {!isLoading && (
              <div className="shrink-0 text-right text-xs text-muted-foreground max-w-[48%] leading-snug">
                {isError ? 'Unavailable' : sub}
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </motion.div>
  )
}

export function IsItWorking() {
  const activation = useActivationMetrics()
  const mission = useMissionSuccessRate()
  const errors = useErrorsBySubsystem('24h')
  const widget = useWidgetEngagement('7d')

  // --- Activation ---
  const act = activation.data
  const activationValue = act && act.total_workspaces > 0 ? `${Math.round(act.rate * 100)}%` : '0%'
  const activationSub = act && act.total_workspaces > 0
    ? `${act.activated}/${act.total_workspaces} workspaces`
    : 'No workspaces yet'

  // --- Mission success rate ---
  const m = mission.data
  const hasMissions = !!m && m.total_executions > 0
  const missionValue = hasMissions ? `${Math.round(m!.value)}%` : '0%'
  const missionSub = hasMissions
    ? `${m!.successful_executions}/${m!.total_executions} missions`
    : 'No missions yet'

  // --- Error rate by subsystem ---
  const e = errors.data
  const errorTotal = e?.total ?? 0
  const worstSubsystem = e?.by_subsystem?.length
    ? [...e.by_subsystem].sort((a, b) => b.count - a.count)[0]
    : null
  const errorSub = errorTotal > 0 && worstSubsystem
    ? `${worstSubsystem.subsystem} (${worstSubsystem.count}) · 24h`
    : 'No errors · 24h'
  const errorAccent = errorTotal > 0 ? 'text-destructive' : 'text-success'

  // --- Widget engagement ---
  const w = widget.data
  const widgetSessions = w?.sessions ?? 0
  const widgetEvents = w?.by_event_type?.reduce((sum, ev) => sum + ev.count, 0) ?? 0
  const widgetSub = widgetSessions > 0
    ? `${widgetEvents} events · 7d`
    : 'No widget activity'

  return (
    <section>
      <div className="flex items-center gap-2 mb-3">
        <h2 className="text-lg font-semibold">Is it working?</h2>
        <Badge variant="outline" className="text-xs text-brand-primary border-brand-primary/30">
          Live metrics
        </Badge>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-5 gap-4">
        <Tile
          icon={Zap}
          label="Activation"
          value={activationValue}
          sub={activationSub}
          accent="text-brand-primary"
          isLoading={activation.isLoading}
          isError={activation.isError}
          delay={0}
        />
        <Tile
          icon={CheckCircle}
          label="Mission success rate"
          value={missionValue}
          sub={missionSub}
          accent="text-success"
          isLoading={mission.isLoading}
          isError={mission.isError}
          delay={0.05}
        />
        <Tile
          icon={AlertTriangle}
          label="Error rate by subsystem"
          value={errorTotal}
          sub={errorSub}
          accent={errorAccent}
          isLoading={errors.isLoading}
          isError={errors.isError}
          delay={0.1}
        />
        <Tile
          icon={MessageSquare}
          label="Widget engagement"
          value={widgetSessions}
          sub={widgetSub}
          accent="text-info"
          isLoading={widget.isLoading}
          isError={widget.isError}
          delay={0.15}
        />
        <Tile
          icon={Boxes}
          label="Per-primitive health"
          value="—"
          sub="Not yet measured"
          accent="text-muted-foreground"
          muted
          delay={0.2}
        />
      </div>
    </section>
  )
}
