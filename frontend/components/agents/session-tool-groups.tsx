'use client'

/**
 * #942 — the "Session tools" picker of a CLI-runtime agent: the core tools every session has,
 * then one checkbox per tool group the owner can give this agent's sessions (data, graph,
 * documents, playbooks, reports, missions — the backend's list and order).
 *
 * The choice is fixed per agent, so every session of the agent is offered the same list (PRD-245
 * D2's prompt-cache rule); the backend still checks each call against the ticket's workspace.
 */

import type { SessionToolGroups } from '@/types/session-tools'

const PICKER_TITLE = 'Session tools'
const DEFAULT_NOTE = 'Defaults: all on'
const ALWAYS_ON_LABEL = 'Always on:'

interface SessionToolGroupsPickerProps {
  groups: SessionToolGroups
  /** the groups ticked on screen */
  selection: string[]
  /** the owner has changed the boxes since the agent was loaded */
  touched: boolean
  alwaysOn: string[]
  onToggle: (groupId: string, on: boolean) => void
}

function ToolNames({ names }: { names: string[] }) {
  return (
    <>
      {names.map((name, index) => (
        <span key={name}>
          {index > 0 ? ', ' : ''}
          <code className="rounded bg-muted/50 px-1 py-0.5 font-mono">{name}</code>
        </span>
      ))}
    </>
  )
}

export function SessionToolGroupsPicker({ groups, selection, touched, alwaysOn, onToggle }: SessionToolGroupsPickerProps) {
  return (
    <div className="space-y-2 rounded-md border border-border/40 p-3 text-xs" data-testid="session-tool-groups">
      <div className="flex items-center justify-between gap-2">
        <p className="font-medium text-foreground">{PICKER_TITLE}</p>
        {groups.is_default && !touched && (
          <span className="text-muted-foreground" data-testid="session-tool-groups-default">
            {DEFAULT_NOTE}
          </span>
        )}
      </div>
      <p className="text-muted-foreground" data-testid="session-tools-always-on">
        {ALWAYS_ON_LABEL} <ToolNames names={alwaysOn} />
      </p>
      <ul className="space-y-2">
        {groups.available.map((group) => (
          <li key={group.id}>
            <label className="flex items-start gap-2">
              <input
                type="checkbox"
                className="mt-0.5"
                aria-label={group.label}
                checked={selection.includes(group.id)}
                onChange={(e) => onToggle(group.id, e.target.checked)}
              />
              <span className="space-y-0.5">
                <span className="block font-medium text-foreground">{group.label}</span>
                {group.description && <span className="block text-muted-foreground">{group.description}</span>}
                {group.tools.length > 0 && (
                  <span className="block text-muted-foreground">
                    <ToolNames names={group.tools} />
                  </span>
                )}
              </span>
            </label>
          </li>
        ))}
      </ul>
    </div>
  )
}

export default SessionToolGroupsPicker
