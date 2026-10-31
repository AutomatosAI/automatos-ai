'use client'

/**
 * PRD-266 US-019b (§4.3 "Web", K15) — a device's push state in Settings → Phones & devices.
 *
 * The row's facts line already says Push on or Push off (device-list.tsx); this adds, when push is
 * off for a reason the server recorded, that reason in words, and on the caller's OWN active phone
 * (iOS or Android: a paired browser gets no pushes) a switch that calls PATCH
 * /api/v1/mobile/devices/{id} {push_enabled}. A phone that never registered an Expo token refuses
 * push on (409 push_token_missing): the app has to open on it first, and the row says so. Another
 * person's row (it carries owner_label) shows the state only: PATCH is the device's person only.
 */
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { isPushTokenMissing, useSetDevicePush, type MobileDevice } from '@/hooks/use-mobile-devices'

const ACTIVE = 'active'
/** The platforms Expo pushes to; a paired browser ('web') has no switch. */
const PUSH_PLATFORMS: ReadonlySet<string> = new Set(['ios', 'android'])
/** Why push is off, as push_disabled_reason records it (services/push/fanout.py, services/mobile/devices.py). */
export const PUSH_OFF_REASONS: Readonly<Record<string, string>> = {
  DeviceNotRegistered: 'The phone no longer has the app (Expo reported it)',
  token_moved: 'Moved to this phone’s newer sign-in',
  device_revoked: 'Revoked',
}
export const PUSH_SWITCH_TEXT = 'Notifications'
export const PUSH_TOKEN_MISSING_LINE = 'Open Automatos Go on this phone to turn notifications on'
export const PUSH_CHANGE_FAILED = 'The switch didn’t change. Try again in a moment.'

/** The caller's own (no owner_label), active phone: the one row kind with a push switch. */
export function hasPushSwitch(device: MobileDevice): boolean {
  return !device.owner_label && device.status === ACTIVE && PUSH_PLATFORMS.has(device.platform ?? '')
}

/** Why push is off, in words; null while it is on or the reason is not one Studio knows. */
export function pushOffReason(device: MobileDevice): string | null {
  if (device.push_enabled || !device.push_disabled_reason) return null
  return PUSH_OFF_REASONS[device.push_disabled_reason] ?? null
}

interface DevicePushStateProps {
  readonly device: MobileDevice
  /** The name the row shows, so the switch says whose notifications it turns. */
  readonly name: string
}

export function DevicePushState({ device, name }: DevicePushStateProps) {
  const setPush = useSetDevicePush()
  const reason = pushOffReason(device)
  const switchable = hasPushSwitch(device)
  if (!reason && !switchable) return null
  const switchId = `device-push-${device.id}`
  return (
    <div className="space-y-1">
      {reason && <p className="text-xs text-muted-foreground">{reason}</p>}
      {switchable && (
        <div className="flex items-center gap-2">
          <Switch
            id={switchId}
            aria-label={`${PUSH_SWITCH_TEXT} on ${name}`}
            checked={device.push_enabled}
            disabled={setPush.isLoading}
            onCheckedChange={(pushEnabled) => setPush.mutate({ deviceId: device.id, pushEnabled })}
          />
          <Label htmlFor={switchId} className="text-xs text-muted-foreground">
            {PUSH_SWITCH_TEXT}
          </Label>
        </div>
      )}
      {setPush.isError && (
        <p role="alert" className="text-xs text-destructive">
          {isPushTokenMissing(setPush.error) ? PUSH_TOKEN_MISSING_LINE : PUSH_CHANGE_FAILED}
        </p>
      )}
    </div>
  )
}
