'use client'

/**
 * PRD-261 A-07d — a pinned version per install (A-07e): a pinned install refuses an update or a
 * rollback to any other version (A-07b), and the server never moves it to a new Marketplace version
 * by itself (an unpinned one it does: services/themes/auto_update.py). Each install offers the
 * versions the server says it can pin (`pinnableVersions`: its current, pending and previous ones)
 * and "Not pinned".
 */
import type { ThemePinState } from '@/lib/api-client'

import { PINS_NOTE } from './theme-labels'

const NOT_PINNED = ''
const SELECT_CLASS = 'rounded-md border border-border bg-background px-2 py-1 text-sm text-foreground'

interface ThemePinsProps {
  readonly pins: readonly ThemePinState[]
  readonly disabled: boolean
  readonly onPin: (pin: { installId: string; version: string | null }) => void
}

export function ThemePins({ pins, disabled, onPin }: ThemePinsProps) {
  if (pins.length === 0) return null
  return (
    <div className="space-y-2 rounded-xl border border-border/50 px-4 py-3">
      <p className="font-medium text-sm text-foreground">Pinned versions</p>
      <p className="text-xs text-muted-foreground">{PINS_NOTE}</p>
      <ul className="space-y-2">
        {pins.map((pin) => (
          <li key={pin.installId} className="flex items-center justify-between gap-3 text-sm">
            <label htmlFor={`theme-pin-${pin.installId}`} className="truncate font-mono text-xs">
              {pin.themeId}
            </label>
            <select
              id={`theme-pin-${pin.installId}`}
              className={SELECT_CLASS}
              value={pin.pinnedVersion ?? NOT_PINNED}
              disabled={disabled}
              onChange={(event) => onPin({ installId: pin.installId, version: event.target.value || null })}
            >
              <option value={NOT_PINNED}>Not pinned</option>
              {pin.pinnableVersions.map((version) => (
                <option key={version} value={version}>
                  {version}
                </option>
              ))}
            </select>
          </li>
        ))}
      </ul>
    </div>
  )
}
