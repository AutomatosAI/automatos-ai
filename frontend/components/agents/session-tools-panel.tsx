'use client'

/**
 * #942 — the session-tool part of a CLI agent's Runtime section: the workspace gaps first (the
 * thing to read before switching), then the group picker.
 *
 * The owner's edit rides the form as `cli_session_tool_groups` (undefined until they change a
 * box), so the existing save sends `configuration.session_tool_groups` only when they did and an
 * untouched agent keeps "absent = every group".
 */

import type { SessionToolGapEntry } from '@/types/session-tools'
import type { RuntimeFields, SessionTool } from './runtime-section'
import { SessionCapabilityGaps } from './session-capability-gaps'
import { SessionToolGroupsPicker } from './session-tool-groups'
import { alwaysOnTools, toggleGroup, workspaceGaps } from './session-tool-groups-model'
import type { SessionToolPreview } from './use-session-tool-preview'

interface SessionToolsPanelProps {
  preview: SessionToolPreview
  value: RuntimeFields
  onChange: (field: 'cli_session_tool_groups', value: string[]) => void
  /** the session-mode settings' tool list (`GET /api/v1/cli-hosts/settings`) */
  sessionTools: SessionTool[]
  /** the saved agent's gaps, from the caller's own fetch, while the preview has not answered */
  savedGaps?: SessionToolGapEntry[] | null
}

export function SessionToolsPanel({ preview, value, onChange, sessionTools, savedGaps }: SessionToolsPanelProps) {
  const { groups } = preview
  const touched = value.cli_session_tool_groups !== undefined
  const selection = value.cli_session_tool_groups ?? groups?.enabled ?? []
  const gaps = workspaceGaps(preview.gaps ?? savedGaps)
  const setGroup = (id: string, on: boolean) => {
    if (groups) onChange('cli_session_tool_groups', toggleGroup(groups, selection, id, on))
  }
  return (
    <>
      <SessionCapabilityGaps gaps={gaps} groups={groups} selection={selection} onEnable={(id) => setGroup(id, true)} />
      {groups && (
        <SessionToolGroupsPicker
          groups={groups}
          selection={selection}
          touched={touched}
          alwaysOn={alwaysOnTools(sessionTools.map((t) => t.name), groups)}
          onToggle={setGroup}
        />
      )}
    </>
  )
}

export default SessionToolsPanel
