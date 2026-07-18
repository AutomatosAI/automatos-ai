'use client'

/**
 * MicHealthControl — the microphone picker + the silent-capture banner
 * (PRD-207).
 *
 * The invisible failure class this kills: the SDK binds a capture device
 * that delivers flat silence (a continuity iPhone, a recorder extension's
 * virtual input, an unplugged USB mic) while the strip says "Mic live" —
 * the caller talks into a dead channel and Auto never hears a word. The
 * hook's analyser measures the user's RMS; when a full window stays at
 * digital silence the banner names the fault and offers the fix — pick
 * another device — right where the eye already is.
 */

import type { ReactNode } from 'react'
import { ChevronDown, MicOff } from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import type { MicInputDevice } from '@/hooks/use-retell-call'

interface DevicePickProps {
  devices: MicInputDevice[]
  /** null = the browser default device. */
  activeDeviceId: string | null
  onPickDevice: (deviceId: string) => void
}

function DeviceMenu({
  devices,
  activeDeviceId,
  onPickDevice,
  trigger,
}: DevicePickProps & { trigger: ReactNode }) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>{trigger}</DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="max-w-[300px]">
        <DropdownMenuLabel className="text-xs">Microphone</DropdownMenuLabel>
        {devices.map((device) => (
          <DropdownMenuCheckboxItem
            key={device.deviceId}
            className="text-xs"
            checked={
              device.deviceId === activeDeviceId ||
              (!activeDeviceId && device.deviceId === 'default')
            }
            onCheckedChange={() => onPickDevice(device.deviceId)}
          >
            <span className="truncate">{device.label}</span>
          </DropdownMenuCheckboxItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

/** The whisper-strip device chooser — present whenever there is a choice. */
export function MicDevicePicker(props: DevicePickProps) {
  if (props.devices.length < 2) return null
  return (
    <DeviceMenu
      {...props}
      trigger={
        <Button
          variant="ghost"
          size="icon"
          className="h-7 w-7 rounded-full"
          aria-label="Choose microphone"
        >
          <ChevronDown className="h-3.5 w-3.5" />
        </Button>
      }
    />
  )
}

/** Shown only when the capture has delivered a full window of silence. */
export function MicSilentBanner(props: DevicePickProps) {
  return (
    <div
      data-testid="mic-silent-banner"
      className="mx-auto mt-1 flex w-fit max-w-full items-center gap-2 rounded-lg border border-warning/40 bg-warning/10 px-3 py-1.5"
    >
      <MicOff className="h-3.5 w-3.5 shrink-0 text-warning" />
      <p className="text-xs text-warning">
        Your microphone is silent — Auto can&apos;t hear you.
      </p>
      {props.devices.length > 1 && (
        <DeviceMenu
          {...props}
          trigger={
            <Button
              size="sm"
              variant="outline"
              className="h-6 border-warning/40 px-2 text-[11px]"
            >
              Pick another mic
            </Button>
          }
        />
      )}
    </div>
  )
}
