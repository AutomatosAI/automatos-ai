'use client'

import * as React from 'react'
import { useFormContext } from 'react-hook-form'
import { Label } from '@/components/ui/label'
import { Input } from '@/components/ui/input'
import type { RecipeFormValues } from './create-recipe-modal'

const MEMORY_ISOLATION_OPTIONS = [
  { value: 'shared', label: 'Shared' },
  { value: 'isolated', label: 'Isolated' },
]

function formatMs(ms: number): string {
  if (ms < 1000) return `${ms}ms`
  if (ms < 60000) return `${(ms / 1000).toFixed(0)}s`
  return `${(ms / 60000).toFixed(1)}min`
}

export function RecipeExecutionConfig() {
  const methods = useFormContext<RecipeFormValues>()
  const config = methods.watch('execution_config')
  const steps = methods.watch('steps')

  const updateConfig = (field: keyof RecipeFormValues['execution_config'], value: string | number | boolean) => {
    methods.setValue('execution_config', { ...config, [field]: value })
  }

  const calculatedMaxTime = React.useMemo(() => {
    const stepCount = Math.max(steps.length, 1)
    return config.mode === 'sequential'
      ? stepCount * config.timeout_per_step
      : config.timeout_per_step
  }, [steps.length, config.mode, config.timeout_per_step])

  return (
    <div className="space-y-4">
      {/* Performance Section */}
      <div className="glass-card rounded-2xl p-5 space-y-4 border border-border/20">
        <h4 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider flex items-center gap-2">
          <span className="text-base">⚡</span> Performance
        </h4>

        {/* Execution Mode */}
        <div>
          <Label className="text-xs text-muted-foreground">Execution Mode</Label>
          <div className="grid grid-cols-2 gap-3 mt-2">
            <button
              type="button"
              onClick={() => updateConfig('mode', 'sequential')}
              className={`p-3 rounded-xl border text-left text-sm transition-all duration-200 ${
                config.mode === 'sequential'
                  ? 'border-primary bg-primary/10 text-foreground'
                  : 'border-border/30 bg-secondary/50 text-muted-foreground hover:border-border/60'
              }`}
            >
              <div className="font-medium">Sequential</div>
              <div className="text-xs mt-0.5 opacity-70">Steps run one after another</div>
            </button>
            <button
              type="button"
              onClick={() => updateConfig('mode', 'parallel')}
              className={`p-3 rounded-xl border text-left text-sm transition-all duration-200 ${
                config.mode === 'parallel'
                  ? 'border-primary bg-primary/10 text-foreground'
                  : 'border-border/30 bg-secondary/50 text-muted-foreground hover:border-border/60'
              }`}
            >
              <div className="font-medium">Parallel</div>
              <div className="text-xs mt-0.5 opacity-70">Steps run simultaneously</div>
            </button>
          </div>
          {config.mode === 'parallel' && (
            <div className="flex items-start gap-2 p-2.5 mt-2 rounded-xl bg-[hsl(var(--warning))]/10 border border-[hsl(var(--warning))]/20 text-xs text-[hsl(var(--warning))]">
              <span className="mt-0.5">⚠</span>
              <span>Parallel mode requires steps to be independent. Ensure no step depends on another&apos;s output.</span>
            </div>
          )}
        </div>
      </div>

      {/* Retry & Timeout Section */}
      <div className="glass-card rounded-2xl p-5 space-y-4 border border-border/20">
        <h4 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider flex items-center gap-2">
          <span className="text-base">🔄</span> Retries & Timeouts
        </h4>

        <div className="grid grid-cols-3 gap-3">
          {/* Max Retries */}
          <div>
            <Label className="text-xs text-muted-foreground">Max Retries per Step</Label>
            <select
              value={config.max_retries}
              onChange={(e) => updateConfig('max_retries', Number(e.target.value))}
              className="w-full mt-1 bg-secondary/50 rounded-xl border border-border/30 px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-primary/50 appearance-none cursor-pointer"
            >
              {Array.from({ length: 6 }, (_, i) => (
                <option key={i} value={i}>{i}</option>
              ))}
            </select>
          </div>

          {/* Per-Step Timeout */}
          <div>
            <Label className="text-xs text-muted-foreground">Step Timeout</Label>
            <Input
              type="number"
              min={10000}
              max={600000}
              step={10000}
              value={config.timeout_per_step}
              onChange={(e) => updateConfig('timeout_per_step', Number(e.target.value))}
              className="mt-1 bg-secondary/50 rounded-xl"
            />
            <span className="text-xs text-muted-foreground/70 mt-1 block">
              {formatMs(config.timeout_per_step)} per step
            </span>
          </div>

          {/* Total Timeout */}
          <div>
            <Label className="text-xs text-muted-foreground">Total Timeout</Label>
            <Input
              type="number"
              min={10000}
              max={3600000}
              step={10000}
              value={config.total_timeout}
              onChange={(e) => updateConfig('total_timeout', Number(e.target.value))}
              className="mt-1 bg-secondary/50 rounded-xl"
            />
            <span className="text-xs text-muted-foreground/70 mt-1 block">
              {formatMs(config.total_timeout)} total
            </span>
          </div>
        </div>

        <div className="p-2.5 rounded-xl bg-secondary/30 border border-border/20 text-xs text-muted-foreground">
          Calculated max time for {steps.length || 1} step{steps.length !== 1 ? 's' : ''} ({config.mode}): <span className="text-primary font-medium">{formatMs(calculatedMaxTime)}</span>
        </div>
      </div>

      {/* Auto-Learning Section */}
      <div className="glass-card rounded-2xl p-5 space-y-4 border border-border/20">
        <h4 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider flex items-center gap-2">
          <span className="text-base">🧠</span> Learning
        </h4>

        {/* Auto-Learning Toggle */}
        <div className="flex items-center justify-between">
          <div>
            <Label className="text-xs text-muted-foreground">Auto-Learning</Label>
            <p className="text-xs text-muted-foreground/60 mt-0.5">Automatically analyze executions for improvement patterns</p>
          </div>
          <button
            type="button"
            onClick={() => updateConfig('auto_learning', !config.auto_learning)}
            className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors duration-200 ${
              config.auto_learning ? 'bg-primary' : 'bg-secondary'
            }`}
          >
            <span
              className={`inline-block h-4 w-4 rounded-full bg-white transition-transform duration-200 ${
                config.auto_learning ? 'translate-x-6' : 'translate-x-1'
              }`}
            />
          </button>
        </div>
      </div>

      {/* Advanced Section */}
      <AdvancedSection config={config} updateConfig={updateConfig} />
    </div>
  )
}

function AdvancedSection({
  config,
  updateConfig,
}: {
  config: RecipeFormValues['execution_config']
  updateConfig: (field: keyof RecipeFormValues['execution_config'], value: string | number | boolean) => void
}) {
  const [expanded, setExpanded] = React.useState(false)

  return (
    <div className="glass-card rounded-2xl border border-border/20 overflow-hidden">
      <button
        type="button"
        onClick={() => setExpanded(!expanded)}
        className="w-full p-4 flex items-center justify-between text-left hover:bg-secondary/30 transition-colors"
      >
        <h4 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">
          Advanced Settings
        </h4>
        <span className="text-muted-foreground text-xs">{expanded ? '▲' : '▼'}</span>
      </button>

      {expanded && (
        <div className="px-5 pb-5 space-y-4 border-t border-border/20 pt-4">
          {/* Parallel Execution Limit */}
          <div>
            <Label className="text-xs text-muted-foreground">Parallel Execution Limit</Label>
            <Input
              type="number"
              min={1}
              max={20}
              value={config.parallel_limit}
              onChange={(e) => updateConfig('parallel_limit', Number(e.target.value))}
              className="mt-1 bg-secondary/50 rounded-xl"
            />
            <span className="text-xs text-muted-foreground/60 mt-1 block">
              Max concurrent steps when running in parallel mode
            </span>
          </div>

          {/* Memory Isolation */}
          <div>
            <Label className="text-xs text-muted-foreground">Memory Isolation</Label>
            <div className="grid grid-cols-2 gap-3 mt-2">
              {MEMORY_ISOLATION_OPTIONS.map((opt) => (
                <button
                  key={opt.value}
                  type="button"
                  onClick={() => updateConfig('memory_isolation', opt.value)}
                  className={`p-3 rounded-xl border text-left text-sm transition-all duration-200 ${
                    config.memory_isolation === opt.value
                      ? 'border-primary bg-primary/10 text-foreground'
                      : 'border-border/30 bg-secondary/50 text-muted-foreground hover:border-border/60'
                  }`}
                >
                  <div className="font-medium">{opt.label}</div>
                  <div className="text-xs mt-0.5 opacity-70">
                    {opt.value === 'shared'
                      ? 'Steps share execution context'
                      : 'Each step has its own context'}
                  </div>
                </button>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
