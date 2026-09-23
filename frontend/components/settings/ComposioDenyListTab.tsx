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
 */

import React, { useEffect, useMemo, useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import { Badge } from '@/components/ui/badge'
import { Textarea } from '@/components/ui/textarea'
import { Loader2, Save, ShieldBan } from 'lucide-react'
import { SystemSetting } from '@/lib/api/system-settings'

export const COMPOSIO_DENIED_ACTIONS_KEY = 'denied_actions'

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
  const [text, setText] = useState((stored ?? []).join('\n'))

  useEffect(() => {
    setText((stored ?? []).join('\n'))
  }, [stored])

  const slugs = normalizeDeniedActions(text)

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
            until it is saved again.
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
            onChange={(event) => setText(event.target.value)}
          />
          <p className="text-xs text-muted-foreground">Matched case-insensitively against the action slug.</p>
        </div>

        <div className="flex items-center gap-2">
          <Button
            onClick={() => onSave({ [COMPOSIO_DENIED_ACTIONS_KEY]: JSON.stringify(slugs) })}
            disabled={saving}
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
      </CardContent>
    </Card>
  )
}
