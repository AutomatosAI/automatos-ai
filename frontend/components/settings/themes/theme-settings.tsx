'use client'

/**
 * PRD-261 A-07d — the workspace's theme settings (A-07e, PRD §5 admin controls, D1-b):
 * private code uploads, members may choose, who may install, and a pinned version per install
 * (theme-pins.tsx). Shown to `workspace:manage` only; every change is one
 * `PUT /api/themes/settings` with the key that changed (manage + step-up). The server refuses what
 * the caller may not change (the platform `off` ceiling, an admin making installs owner-only) with
 * a sentence the hook shows.
 */
import type { ReactNode } from 'react'

import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { usePutThemeSettings } from '@/hooks/use-installed-themes'
import type { ThemeSettings, ThemeSettingsChange } from '@/lib/api-client'

import { ThemePins } from './theme-pins'

export const INSTALL_ROLE_LABELS: Readonly<Record<string, string>> = {
  admins: 'Owners and admins',
  owner: 'The owner only',
}

const SELECT_CLASS = 'rounded-md border border-border bg-background px-2 py-1 text-sm text-foreground'

function SettingRow({ id, label, hint, children }: { id: string; label: string; hint: string; children: ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-4 rounded-xl border border-border/50 px-4 py-3">
      <div>
        <Label htmlFor={id} className="font-medium">
          {label}
        </Label>
        <p className="mt-0.5 text-xs text-muted-foreground">{hint}</p>
      </div>
      {children}
    </div>
  )
}

function privateUploadsHint(settings: ThemeSettings): string {
  const { privateUploads } = settings
  if (privateUploads.platform === 'off') return 'Switched off for the whole platform; a workspace can’t turn it on.'
  const from = privateUploads.source === 'default' ? `the default (${privateUploads.default})` : 'set for this workspace'
  return `Lets owners and admins upload theme code nobody reviewed. Now ${privateUploads.effective}, from ${from}.`
}

export function ThemeSettingsCard({ settings }: { settings: ThemeSettings }) {
  const put = usePutThemeSettings()
  const save = (changes: ThemeSettingsChange) => put.mutate(changes)
  const uploadsOn = settings.privateUploads.effective !== 'off'

  return (
    <section aria-label="Theme settings" className="space-y-2" data-testid="theme-settings">
      <SettingRow id="theme-private-uploads" label="Private theme uploads" hint={privateUploadsHint(settings)}>
        <Switch
          id="theme-private-uploads"
          checked={uploadsOn}
          disabled={put.isLoading || settings.privateUploads.platform === 'off'}
          onCheckedChange={(on) => save({ privateUploads: on ? 'on' : 'off' })}
        />
      </SettingRow>
      <SettingRow
        id="theme-members-may-choose"
        label="Members choose their own theme"
        hint="Off: everyone sees the workspace default."
      >
        <Switch
          id="theme-members-may-choose"
          checked={settings.membersMayChoose}
          disabled={put.isLoading}
          onCheckedChange={(on) => save({ membersMayChoose: on })}
        />
      </SettingRow>
      <SettingRow id="theme-install-role" label="Who may install themes" hint="Install, update, roll back and pin.">
        <select
          id="theme-install-role"
          className={SELECT_CLASS}
          value={settings.installRole}
          disabled={put.isLoading}
          onChange={(event) => save({ installRole: event.target.value })}
        >
          {settings.installRoles.map((role) => (
            <option key={role} value={role}>
              {INSTALL_ROLE_LABELS[role] ?? role}
            </option>
          ))}
        </select>
      </SettingRow>
      <ThemePins pins={settings.pins} disabled={put.isLoading} onPin={(pin) => save({ pins: [pin] })} />
    </section>
  )
}
