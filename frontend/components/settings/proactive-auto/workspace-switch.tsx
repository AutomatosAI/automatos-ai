'use client'

/**
 * PRD-268 §6.9 — the workspace switch, `settings.proactive.enabled`.
 *
 * - An owner or admin flips it: PUT /api/proactive/settings `{enabled}`, shown at once and put back, with the
 *   server's sentence, when the server refuses (useChangeProactiveSettings).
 * - Anyone else sees it read-only, from the workspace they already load (GET /api/workspaces/current answers its
 *   settings; GET /api/proactive/settings is the owners' and admins'), with the line saying who can change it.
 * - Under it, one line on what Auto is doing here, from the workspace's `effective_mode` (GET /api/proactive/settings
 *   for an owner or admin, GET /api/proactive/me for anyone else): nothing when it is off and so is the switch.
 */
import { useChangeProactiveSettings, useMyProactiveSettings } from '@/hooks/use-proactive-settings'
import { workspaceSwitchOf, type ProactiveSettings } from '@/lib/proactive/settings'
import { ASK_AN_OWNER_OR_ADMIN, WORKSPACE_OFF, WORKSPACE_ON, WORKSPACE_SWITCH_LABEL } from './copy'
import { FailureLine, ModeLine, SwitchRow } from './parts'

/** The switch an owner or admin flips, and what Auto is doing here (the settings read's mode). */
export function ManagedWorkspaceSwitch({ settings }: { readonly settings: ProactiveSettings }) {
  const change = useChangeProactiveSettings()
  const { enabled } = settings
  return (
    <div>
      <SwitchRow
        label={WORKSPACE_SWITCH_LABEL}
        detail={enabled ? WORKSPACE_ON : WORKSPACE_OFF}
        checked={enabled}
        disabled={change.isLoading}
        onCheckedChange={(on) => change.mutate({ enabled: on })}
      />
      <ModeLine mode={settings.effective_mode} switchedOn={enabled} />
      <FailureLine error={change.isError ? change.error : null} />
    </div>
  )
}

/** The switch as everyone else sees it: read-only, who can change it, and what Auto is doing (their own read's mode). */
export function ReadOnlyWorkspaceSwitch({ settings }: { readonly settings: unknown }) {
  const enabled = workspaceSwitchOf(settings)
  const mode = useMyProactiveSettings().data?.effective_mode ?? null
  return (
    <div>
      <SwitchRow
        label={WORKSPACE_SWITCH_LABEL}
        detail={enabled ? WORKSPACE_ON : WORKSPACE_OFF}
        checked={enabled}
        disabled
      />
      <ModeLine mode={mode} switchedOn={enabled} />
      <p className="text-xs text-muted-foreground">{ASK_AN_OWNER_OR_ADMIN}</p>
    </div>
  )
}
