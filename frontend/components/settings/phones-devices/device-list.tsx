'use client'

/**
 * PRD-266 US-005c — the device list in Settings → Phones & devices (§4.3 "Web"): every phone or
 * paired browser GET /api/v1/mobile/devices returns (the caller's own; for a workspace owner or
 * admin, and the local operator, the workspace's) with its platform, when it last called the API
 * and whether push is on. Revoke asks first, then calls DELETE /devices/{id}; a revoked device
 * stays listed as revoked. Revoking this browser's own pairing sends it back to "Pair this
 * browser" on its next call (device_revoked).
 *
 * US-019b (K15): an owner or admin sees whose each phone is (owner_label, beside its name: a hosted
 * phone's name is the model the app registers); each row's push state, why it is off, and the push
 * switch on the caller's own phones are device-push-state.tsx.
 */
import { useState } from 'react'
import { Loader2 } from 'lucide-react'

import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { useRevokeDevice, type MobileDevice } from '@/hooks/use-mobile-devices'
import { formatTimestamp } from '@/lib/utils'

import { DevicePushState } from './device-push-state'

const ACTIVE = 'active'
const PLATFORM_LABELS: Readonly<Record<string, string>> = { ios: 'iOS', android: 'Android', web: 'Browser' }
/** What Revoke does to each kind of device (services/mobile/devices.py revoke_device). */
const REVOKE_EFFECT: Readonly<Record<string, string>> = {
  paired: 'Its next call is refused; to come back, it pairs again with a new code.',
  clerk: 'It stops getting this workspace’s notifications on that phone. Its account sign-in is not changed.',
}
const REVOKE_EFFECT_OTHER = 'It is cut off from this workspace.'
export const UNNAMED_DEVICE = 'Unnamed device'
export const NO_DEVICES = 'No phones or paired browsers yet.'
export const LIST_FAILED = 'The devices could not be loaded. Try again in a moment.'
export const REVOKE_FAILED = 'Revoking didn’t go through. Try again in a moment.'

function deviceName(device: MobileDevice): string {
  return device.name || UNNAMED_DEVICE
}

/** "iOS · Last seen 5 minutes ago · Push on" */
function deviceFacts(device: MobileDevice): string {
  const platform = (device.platform && PLATFORM_LABELS[device.platform]) || 'Device'
  const seen = device.last_seen_at ? `Last seen ${formatTimestamp(device.last_seen_at)}` : 'Not seen yet'
  return `${platform} · ${seen} · ${device.push_enabled ? 'Push on' : 'Push off'}`
}

function DeviceRow({ device, onRevoke }: { device: MobileDevice; onRevoke: (device: MobileDevice) => void }) {
  return (
    <li className="flex flex-col gap-2 rounded-lg border border-border/40 px-3 py-2 text-sm sm:flex-row sm:items-center sm:justify-between">
      <div className="min-w-0 space-y-0.5">
        <div className="flex min-w-0 items-baseline gap-2">
          <p className="truncate font-medium text-foreground">{deviceName(device)}</p>
          {device.owner_label && <span className="truncate text-xs text-muted-foreground">{device.owner_label}</span>}
        </div>
        <p className="text-xs text-muted-foreground">{deviceFacts(device)}</p>
        <DevicePushState device={device} name={deviceName(device)} />
      </div>
      {device.status === ACTIVE ? (
        <Button type="button" variant="outline" size="sm" onClick={() => onRevoke(device)} aria-label={`Revoke ${deviceName(device)}`}>
          Revoke
        </Button>
      ) : (
        <span className="text-xs capitalize text-muted-foreground">{device.status}</span>
      )}
    </li>
  )
}

interface RevokeDeviceDialogProps {
  readonly target: MobileDevice | null
  readonly pending: boolean
  readonly failed: boolean
  readonly onClose: () => void
  readonly onConfirm: (device: MobileDevice) => void
}

function RevokeDeviceDialog({ target, pending, failed, onClose, onConfirm }: RevokeDeviceDialogProps) {
  return (
    <Dialog open={target !== null} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="sm:max-w-sm">
        <DialogHeader>
          <DialogTitle>Revoke {target ? deviceName(target) : 'this device'}?</DialogTitle>
          <DialogDescription>{target ? REVOKE_EFFECT[target.kind] ?? REVOKE_EFFECT_OTHER : ''}</DialogDescription>
        </DialogHeader>
        {failed && <p role="alert" className="text-sm text-destructive">{REVOKE_FAILED}</p>}
        <DialogFooter>
          <Button type="button" variant="outline" onClick={onClose}>
            Cancel
          </Button>
          <Button type="button" variant="destructive" disabled={pending} onClick={() => target && onConfirm(target)}>
            {pending && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
            Revoke
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

interface DeviceListProps {
  readonly devices: readonly MobileDevice[] | undefined
  readonly loading: boolean
  readonly failed: boolean
}

function DeviceListBody({ devices, loading, failed, onRevoke }: DeviceListProps & { onRevoke: (device: MobileDevice) => void }) {
  if (loading) return <p className="text-sm text-muted-foreground">Loading devices…</p>
  if (failed) return <p role="alert" className="text-sm text-destructive">{LIST_FAILED}</p>
  if (!devices || devices.length === 0) return <p className="text-sm text-muted-foreground">{NO_DEVICES}</p>
  return (
    <ul className="space-y-2">
      {devices.map((device) => (
        <DeviceRow key={device.id} device={device} onRevoke={onRevoke} />
      ))}
    </ul>
  )
}

export function DeviceList({ devices, loading, failed }: DeviceListProps) {
  const revoke = useRevokeDevice()
  const [target, setTarget] = useState<MobileDevice | null>(null)
  const close = () => {
    setTarget(null)
    revoke.reset()
  }
  const confirm = (device: MobileDevice) => revoke.mutate(device.id, { onSuccess: () => setTarget(null) })

  return (
    <section aria-labelledby="phones-devices-list-title" className="space-y-2">
      <h3 id="phones-devices-list-title" className="text-xs uppercase tracking-wide text-muted-foreground">
        Devices
      </h3>
      <DeviceListBody devices={devices} loading={loading} failed={failed} onRevoke={setTarget} />
      <RevokeDeviceDialog
        target={target}
        pending={revoke.isLoading}
        failed={revoke.isError}
        onClose={close}
        onConfirm={confirm}
      />
    </section>
  )
}
