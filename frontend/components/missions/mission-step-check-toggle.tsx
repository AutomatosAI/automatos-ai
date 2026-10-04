'use client'

import { toast } from 'sonner'
import { Label } from '@/components/ui/label'
import { cn } from '@/lib/utils'
import { useMission, useUpdateMissionSettings } from '@/hooks/use-missions-api'
import { TERMINAL_RUN_STATES } from '@/types/missions'

interface MissionStepCheckToggleProps {
  missionId: string
}

/**
 * F282 (night 8): the mission page's half of the one check_each_step switch —
 * the New mission form (create-mission-modal.tsx) sets it at creation, this
 * sets it afterwards. Shows the mission's current setting, normalised
 * server-side (MissionResponse.check_each_step) whatever spelling Auto wrote
 * it under, and is disabled once the mission can no longer run.
 *
 * A standalone component, not a slot inside MissionDetailPage: that component
 * is already well past the file's own size limit, so this mounts alongside
 * it from the route (app/missions/[id]/page.tsx) instead of growing it.
 */
export function MissionStepCheckToggle({ missionId }: MissionStepCheckToggleProps) {
  const { data: mission, isLoading } = useMission(missionId)
  const updateSettings = useUpdateMissionSettings()

  if (isLoading || !mission) return null

  const checked = mission.check_each_step
  const canChange = !(TERMINAL_RUN_STATES as readonly string[]).includes(mission.state)
  const disabled = !canChange || updateSettings.isLoading

  const toggle = () => {
    updateSettings.mutate(
      { id: missionId, body: { check_each_step: !checked } },
      { onError: (err) => toast.error(err.message || 'Failed to update the mission settings') },
    )
  }

  return (
    <div className="mx-4 md:mx-6 mt-4 flex items-center justify-between rounded-lg border border-border px-3 py-2.5">
      <div className="space-y-0.5">
        <Label htmlFor="mission-check-each-step" className={cn('text-sm', !disabled && 'cursor-pointer')}>
          Check each step with me
        </Label>
        <p className="text-[11px] text-muted-foreground">
          {canChange
            ? 'Every step waits in Review for your OK before the next one starts'
            : 'This mission has finished, so the setting can no longer change'}
        </p>
      </div>
      <button
        id="mission-check-each-step"
        type="button"
        role="switch"
        aria-checked={checked}
        onClick={toggle}
        disabled={disabled}
        className={cn(
          'relative inline-flex h-5 w-9 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors',
          'disabled:cursor-not-allowed disabled:opacity-50',
          checked ? 'bg-primary' : 'bg-muted',
        )}
      >
        <span
          className={cn(
            'pointer-events-none inline-block h-4 w-4 rounded-full bg-background shadow-lg ring-0 transition-transform',
            checked ? 'translate-x-4' : 'translate-x-0',
          )}
        />
      </button>
    </div>
  )
}
