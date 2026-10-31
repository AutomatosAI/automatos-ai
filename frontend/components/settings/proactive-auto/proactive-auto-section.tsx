'use client'

/**
 * PRD-268 §6.9, §6.12.4 — Settings → Orchestrator, "Proactive Auto": the switches that today exist only as routes.
 *
 * - **The workspace switch** (`settings.proactive.enabled`): an owner or admin flips it and tightens the caps
 *   (GET/PUT /api/proactive/settings); everyone else sees the switch read-only with who can change it.
 * - **Pause suggestions**: the signed-in person's own, for every role (GET/PUT /api/proactive/me).
 *
 * Neither route answers the platform's mode (off, shadow or live, a system admin's), so the section says nothing
 * about it. Both editions draw the same section: the switches are workspace data, and only who may edit them
 * depends on the role. /settings is a host-owned route (HOST_OWNED_ROUTES), so it looks the same under every theme,
 * built like "What Auto depends on" beside it.
 */
import { CARD_CLASS, ENTRANCE_CLASS, Eyebrow, QUIET_BUTTON_CLASS } from '@/components/proactive/card-parts'
import { type Workspace, useWorkspaceOptional } from '@/components/workspace-provider'
import { useChangeProactiveSettings, useProactiveSettings } from '@/hooks/use-proactive-settings'
import { CAPS_STORED_WRONG, DEFAULT_CAPS_CHANGE, canManageProactive } from '@/lib/proactive/settings'
import { refusalOf } from '@/lib/proactive/wire'
import { cn } from '@/lib/utils'
import { CapsForm } from './caps-form'
import { LOADING, PROACTIVE_INTRO, PROACTIVE_TITLE, RESET_CAPS, SAVING } from './copy'
import { FailureLine } from './parts'
import { PauseSuggestionsRow } from './pause-row'
import { ManagedWorkspaceSwitch, ReadOnlyWorkspaceSwitch } from './workspace-switch'

function Loading() {
  return (
    <p role="status" className="text-xs text-muted-foreground">
      {LOADING}
    </p>
  )
}

/** Caps stored wrong (409 `caps_stored_wrong`) refuse every check-in until saved again: this saves the defaults. */
function ResetStoredCaps() {
  const change = useChangeProactiveSettings()
  return (
    <div className="flex flex-wrap items-center gap-3">
      <button
        type="button"
        disabled={change.isLoading}
        aria-busy={change.isLoading}
        onClick={() => change.mutate(DEFAULT_CAPS_CHANGE)}
        className={QUIET_BUTTON_CLASS}
      >
        {change.isLoading ? SAVING : RESET_CAPS}
      </button>
      <FailureLine error={change.isError ? change.error : null} />
    </div>
  )
}

/** Why the settings couldn't be read, and the way out when it is caps stored wrong. */
function ReadFailure({ error }: { readonly error: unknown }) {
  return (
    <div className="space-y-2">
      <FailureLine error={error} />
      {refusalOf(error)?.code === CAPS_STORED_WRONG ? <ResetStoredCaps /> : null}
    </div>
  )
}

/** An owner's or an admin's: the switch they flip and the caps they tighten, from the settings route. */
function ManagedWorkspace() {
  const settings = useProactiveSettings()
  if (settings.data === undefined) return settings.isError ? <ReadFailure error={settings.error} /> : <Loading />
  return (
    <>
      <ManagedWorkspaceSwitch enabled={settings.data.enabled} />
      <CapsForm settings={settings.data} />
    </>
  )
}

/** The workspace's part: editable for an owner or admin, read-only for anyone else, once the workspace is read. */
function WorkspacePart({ workspace }: { readonly workspace: Workspace | null }) {
  if (workspace === null) return <Loading />
  if (canManageProactive(workspace.role)) return <ManagedWorkspace />
  return <ReadOnlyWorkspaceSwitch settings={workspace.settings} />
}

/** Proactive Auto's workspace switch, its caps for those who may change them, and the person's own pause. */
export function ProactiveAutoSection() {
  const workspace = useWorkspaceOptional()?.workspace ?? null
  return (
    <section aria-label={PROACTIVE_TITLE} className={cn(CARD_CLASS, ENTRANCE_CLASS, 'space-y-3')}>
      <Eyebrow>{PROACTIVE_TITLE}</Eyebrow>
      <p className="text-sm text-muted-foreground">{PROACTIVE_INTRO}</p>
      <WorkspacePart workspace={workspace} />
      <PauseSuggestionsRow />
    </section>
  )
}
