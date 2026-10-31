'use client'

/**
 * PRD-268 §6.9 — the workspace switch, `settings.proactive.enabled`.
 *
 * - An owner or admin flips it: PUT /api/proactive/settings `{enabled}`, shown at once and put back, with the
 *   server's sentence, when the server refuses (useChangeProactiveSettings).
 * - Anyone else sees it read-only, from the workspace they already load (GET /api/workspaces/current answers its
 *   settings; GET /api/proactive/settings is the owners' and admins'), with the line saying who can change it.
 */
import { useChangeProactiveSettings } from '@/hooks/use-proactive-settings'
import { workspaceSwitchOf } from '@/lib/proactive/settings'
import { ASK_AN_OWNER_OR_ADMIN, WORKSPACE_OFF, WORKSPACE_ON, WORKSPACE_SWITCH_LABEL } from './copy'
import { FailureLine, SwitchRow } from './parts'

/** The switch an owner or admin flips. */
export function ManagedWorkspaceSwitch({ enabled }: { readonly enabled: boolean }) {
  const change = useChangeProactiveSettings()
  return (
    <div>
      <SwitchRow
        label={WORKSPACE_SWITCH_LABEL}
        detail={enabled ? WORKSPACE_ON : WORKSPACE_OFF}
        checked={enabled}
        disabled={change.isLoading}
        onCheckedChange={(on) => change.mutate({ enabled: on })}
      />
      <FailureLine error={change.isError ? change.error : null} />
    </div>
  )
}

/** The switch as everyone else sees it: its state, read-only, and who can change it. */
export function ReadOnlyWorkspaceSwitch({ settings }: { readonly settings: unknown }) {
  const enabled = workspaceSwitchOf(settings)
  return (
    <div>
      <SwitchRow
        label={WORKSPACE_SWITCH_LABEL}
        detail={enabled ? WORKSPACE_ON : WORKSPACE_OFF}
        checked={enabled}
        disabled
      />
      <p className="text-xs text-muted-foreground">{ASK_AN_OWNER_OR_ADMIN}</p>
    </div>
  )
}
