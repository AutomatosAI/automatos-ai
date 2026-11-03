'use client'

/**
 * PRD-270 W2 US-270-14e (§5.6 "Updates arrive as new versions with an update notice", K17, K19, K30) — the
 * update notices above the Marketplace's tabs, rendered by CloudItemsLayer (local edition only: the layer
 * mounts nothing on the hosted edition).
 *
 * From `GET /api/marketplace/cloud-sync/updates` (the caller's workspace), one quiet line per item:
 *
 * - a Cloud agent or Playbook this workspace holds copies of, none at the version Automatos Cloud has now:
 *   "<name> has a newer version on Automatos Cloud. Your copy stays as it is." with "Review the new version",
 *   which opens its disclosure through the layer. Its Add marks a one-time allowance at the install gate
 *   (`allowNext`: the disclosure was just read, so the gate doesn't ask again) and installs a NEW copy through
 *   the normal install call; the old copy stays the workspace's to keep or delete. The notice clears once a
 *   copy at the current version exists (K17).
 * - a Cloud plugin this workspace enabled before it moved: enabled by reference, it already runs the new
 *   version (K15). "<name> was updated to <version> on Automatos Cloud; this workspace now runs it." with
 *   Review (its disclosure, no Add) and Dismiss, kept per viewer in localStorage by the workspace, the plugin
 *   and the revision it moved to (every read and write in try/catch: with storage blocked it is hidden for
 *   this page only). A later update, a new revision, shows again.
 *
 * Nothing renders while there is none, while it reads, or when it can't be read (`cloudUpdatesOf`). A Sync
 * reads it again (W1's Sync invalidates the Cloud items root), and so does an agent or Playbook install
 * through its install hook (useInstallMarketplaceItem, useInstallPlaybookFromMarketplace), a notice's Add
 * included. A notice is secondary, never a second ink primary (K30); every name and version is the server's,
 * rendered as text.
 */
import { useState } from 'react'
import { toast } from 'sonner'

import { Button } from '@/components/ui/button'
import { useWorkspace } from '@/components/workspace-provider'
import { useCloudUpdates } from '@/hooks/use-cloud-items'
import { useInstallMarketplaceItem } from '@/hooks/use-marketplace-api'
import { useInstallPlaybookFromMarketplace } from '@/hooks/use-playbook-api'
import type { CloudInstallCall } from '@/lib/marketplace/cloud-install-call'
import { allowNext } from '@/lib/marketplace/cloud-install-gate'
import { isJson } from '@/lib/marketplace/cloud-item-rows'
import type { CloudUpdate } from '@/lib/marketplace/cloud-items'

import {
  DISMISS, INSTALL_WARNING, PLAYBOOK_COPY_ADDED, PLAYBOOK_COPY_FAILED, REVIEW, REVIEW_NEW_VERSION, UPDATES_LABEL,
  copyAddedLine, newerVersionLine, pluginUpdatedLine,
} from './item-copy'

/** Opens a notice's review in the layer; its Add (none for a plugin) resolves true once the new copy is added. */
export type ReviewUpdate = (call: CloudInstallCall, add?: () => Promise<boolean>) => void

/** One viewer's dismissal of one plugin update, by the workspace, the plugin and the revision it moved to. */
export const DISMISSED_KEY_PREFIX = 'automatos:cloud-update-dismissed:'

export function dismissedKey(workspaceId: string | null, update: CloudUpdate): string {
  return `${DISMISSED_KEY_PREFIX}${workspaceId ?? ''}:${String(update.cloudId)}:${update.available.revision}`
}

function readDismissed(key: string): boolean {
  try {
    return window.localStorage.getItem(key) === '1'
  } catch {
    return false
  }
}

function writeDismissed(key: string): void {
  try {
    window.localStorage.setItem(key, '1')
  } catch {
    // Storage blocked: the notice stays hidden for this page only.
  }
}

/** A Playbook install's own toasts, as the Playbooks tab shows them: its message, then each warning. */
function playbookAdded(name: string, answer: unknown): void {
  const message = isJson(answer) && typeof answer.message === 'string' ? answer.message : copyAddedLine(name)
  toast.success(PLAYBOOK_COPY_ADDED, { description: message })
  const warnings = isJson(answer) && Array.isArray(answer.warnings) ? answer.warnings : []
  warnings.filter((warning): warning is string => typeof warning === 'string')
    .forEach((warning) => toast(INSTALL_WARNING, { description: warning }))
}

/**
 * Adds a new copy of a Cloud agent or Playbook through its normal install hook, which the gate lets through
 * once; true once it is added. The agent hook reports its own outcome; a Playbook's is reported here. Both hooks
 * read the notices again once the copy is added.
 */
function useAddNewCopy(): (update: CloudUpdate) => Promise<boolean> {
  const installAgent = useInstallMarketplaceItem()
  const installPlaybook = useInstallPlaybookFromMarketplace()
  return async (update) => {
    const id = Number(update.cloudId)
    allowNext(update.kind, id)
    try {
      if (update.kind === 'agent') await installAgent.mutateAsync(id)
      else playbookAdded(update.name, await installPlaybook.mutateAsync(id))
      return true
    } catch (error) {
      const description = error instanceof Error && error.message ? error.message : undefined
      if (update.kind !== 'agent') toast.error(PLAYBOOK_COPY_FAILED, { description })
      return false
    }
  }
}

interface NoticeLineProps {
  update: CloudUpdate
  onReview: () => void
  onDismiss: () => void
}

function NoticeLine({ update, onReview, onDismiss }: NoticeLineProps) {
  const plugin = update.kind === 'plugin'
  return (
    <li className="flex flex-wrap items-center gap-x-3 gap-y-1" data-testid="cloud-update-notice">
      <span className="min-w-0 break-words">
        {plugin ? pluginUpdatedLine(update.name, update.available.version) : newerVersionLine(update.name)}
      </span>
      <Button type="button" size="sm" variant="outline" onClick={onReview} data-testid="cloud-update-review">
        {plugin ? REVIEW : REVIEW_NEW_VERSION}
      </Button>
      {plugin && (
        <Button type="button" size="sm" variant="ghost" onClick={onDismiss} data-testid="cloud-update-dismiss">
          {DISMISS}
        </Button>
      )}
    </li>
  )
}

/** The workspace's update notices; nothing when there is none. */
export function CloudUpdateNotices({ onReview }: { onReview: ReviewUpdate }) {
  const workspaceId = useWorkspace().workspace?.id ?? null
  const answer = useCloudUpdates(workspaceId)
  const addNewCopy = useAddNewCopy()
  const [dismissed, setDismissed] = useState<readonly string[]>([])
  const hidden = (update: CloudUpdate) => {
    if (update.kind !== 'plugin') return false
    const key = dismissedKey(workspaceId, update)
    return dismissed.includes(key) || readDismissed(key)
  }
  const shown = (answer.data?.updates ?? []).filter((update) => !hidden(update))
  if (shown.length === 0) return null

  const review = (update: CloudUpdate) => {
    const call = { kind: update.kind, key: update.cloudId }
    onReview(call, update.kind === 'plugin' ? undefined : () => addNewCopy(update))
  }
  const dismiss = (update: CloudUpdate) => {
    const key = dismissedKey(workspaceId, update)
    writeDismissed(key)
    setDismissed((held) => [...held, key])
  }
  return (
    <section
      aria-label={UPDATES_LABEL}
      className="rounded-xl border border-border/50 px-4 py-3 text-sm text-muted-foreground"
      data-testid="cloud-update-notices"
    >
      <ul className="space-y-2">
        {shown.map((update) => (
          <NoticeLine
            key={`${update.kind}:${String(update.cloudId)}`}
            update={update}
            onReview={() => review(update)}
            onDismiss={() => dismiss(update)}
          />
        ))}
      </ul>
    </section>
  )
}
