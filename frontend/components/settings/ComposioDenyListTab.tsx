/**
 * Composio deny list — platform-wide (PRD-251 S0.6 / D16, super-admin surface)
 * ============================================================================
 *
 * The `composio.denied_actions` system setting, on the same by-category plane
 * as every other tab: a JSON list of Composio action slugs that no agent,
 * Playbook or API call may run, in any workspace, whatever the policy plane
 * mode. The backend reads it on every call, so a save takes effect on the next
 * action — no restart, no redeploy. One slug per line here; the save writes the
 * JSON list (upper-cased, de-duplicated).
 *
 * The backend refuses every action while the stored value is unreadable, and
 * denies nothing once the list is empty. So this screen never turns the one
 * into the other by accident. An unreadable value is shown verbatim, for
 * repair. A line that is not an action slug blocks Save (saved, it would match
 * nothing). An empty list is saved only after a second, explicit confirmation.
 */

import React, { useEffect, useMemo, useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import { Badge } from '@/components/ui/badge'
import { Textarea } from '@/components/ui/textarea'
import { DeleteConfirmation } from '@/components/shared/delete-confirmation'
import { Loader2, Save, ShieldBan } from 'lucide-react'
import { SystemSetting } from '@/lib/api/system-settings'

export const COMPOSIO_DENIED_ACTIONS_KEY = 'denied_actions'

/** A Composio action slug, upper-cased: letters, digits and underscores. */
export const ACTION_SLUG_PATTERN = /^[A-Z0-9_]+$/

/** What saving an empty list does, said before it is saved. */
export const EMPTY_DENY_LIST_WARNING =
  'With no slugs on the list, every Composio action becomes runnable by any agent, Playbook or API ' +
  'call in every workspace, the Higgsfield billing actions included.'

interface ComposioDenyListTabProps {
  settings: SystemSetting[]
  onSave: (updates: Record<string, string>) => void
  saving: boolean
  onReset: () => void
}

/** The stored JSON list → slugs, or null when it is not a JSON list of strings. */
export function parseDeniedActions(raw: string | null | undefined): string[] | null {
  if (!raw || !raw.trim()) return []
  try {
    const value: unknown = JSON.parse(raw)
    if (Array.isArray(value) && value.every((slug) => typeof slug === 'string')) return value as string[]
  } catch {
    // fall through: unreadable
  }
  return null
}

/** One slug per line → the list the backend reads: trimmed, upper-cased, de-duplicated. */
export function normalizeDeniedActions(text: string): string[] {
  const slugs = text
    .split('\n')
    .map((line) => line.trim().toUpperCase())
    .filter((line) => line.length > 0)
  return Array.from(new Set(slugs))
}

/** The normalised lines that are not action slugs (a pasted JSON list, a stored value not yet rewritten). */
export function notActionSlugs(slugs: string[]): string[] {
  return slugs.filter((slug) => !ACTION_SLUG_PATTERN.test(slug))
}

/** The text the editor starts from: the stored slugs one per line, or an
 * unreadable stored value verbatim, so the admin sees what to repair. */
export function denyListEditorText(raw: string | null | undefined): string {
  const stored = parseDeniedActions(raw)
  return stored === null ? raw ?? '' : stored.join('\n')
}

export default function ComposioDenyListTab({
  settings,
  onSave,
  saving,
  onReset,
}: ComposioDenyListTabProps) {
  const setting = settings.find((s) => s.key === COMPOSIO_DENIED_ACTIONS_KEY)
  // The stored value only: with no value the backend denies nothing, so the
  // default must not be shown as if it applied.
  const stored = useMemo(() => parseDeniedActions(setting?.value), [setting])
  const [text, setText] = useState(() => denyListEditorText(setting?.value))
  const [confirmingEmpty, setConfirmingEmpty] = useState(false)

  useEffect(() => {
    setText(denyListEditorText(setting?.value))
  }, [setting])

  const slugs = normalizeDeniedActions(text)
  const invalid = notActionSlugs(slugs)
  const save = () => onSave({ [COMPOSIO_DENIED_ACTIONS_KEY]: JSON.stringify(slugs) })

  if (!setting) {
    return (
      <Card>
        <CardContent className="py-8 text-center text-sm text-muted-foreground">
          The Composio deny list has not been seeded yet — run the pending migrations.
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <ShieldBan className="h-5 w-5" />
          Composio deny list
          <Badge variant="secondary">
            {slugs.length} {slugs.length === 1 ? 'action' : 'actions'}
          </Badge>
        </CardTitle>
        <CardDescription>
          Composio actions no agent, Playbook or API call may run, in any workspace — whatever the
          policy plane mode. Buying credits, changing plans and deploying stay with a person, in the
          tool&apos;s own interface. A save takes effect on the next call.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {stored === null && (
          <p className="rounded-lg border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive" role="alert">
            The stored list is not a JSON list of action slugs, so every Composio action is refused
            until it is saved again. It is shown below as stored: rewrite it as one action slug per
            line, then save.
          </p>
        )}
        <div className="space-y-1.5">
          <Label htmlFor="composio-denied-actions" className="font-medium">
            Denied action slugs, one per line
          </Label>
          <Textarea
            id="composio-denied-actions"
            value={text}
            rows={10}
            spellCheck={false}
            className="font-mono text-xs"
            aria-invalid={invalid.length > 0}
            aria-describedby={invalid.length > 0 ? 'composio-denied-actions-invalid' : undefined}
            onChange={(event) => setText(event.target.value)}
          />
          {invalid.length > 0 ? (
            <p id="composio-denied-actions-invalid" className="text-xs text-destructive">
              Not action slugs, so the list cannot be saved: {invalid.join(', ')}. Write one slug per
              line, in letters, digits and underscores.
            </p>
          ) : (
            <p className="text-xs text-muted-foreground">Matched case-insensitively against the action slug.</p>
          )}
        </div>

        <div className="flex items-center gap-2">
          <Button
            onClick={() => (slugs.length === 0 ? setConfirmingEmpty(true) : save())}
            disabled={saving || invalid.length > 0}
          >
            {saving ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" /> Saving…
              </>
            ) : (
              <>
                <Save className="w-4 h-4 mr-2" /> Save deny list
              </>
            )}
          </Button>
          <Button variant="outline" onClick={onReset} disabled={saving}>
            Reset to defaults
          </Button>
        </div>

        <DeleteConfirmation
          open={confirmingEmpty}
          onOpenChange={setConfirmingEmpty}
          title="Save an empty deny list?"
          description={EMPTY_DENY_LIST_WARNING}
          confirmLabel="Save the empty list"
          onConfirm={save}
        />
      </CardContent>
    </Card>
  )
}
