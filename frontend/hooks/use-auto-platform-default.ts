/**
 * The "Platform default for Auto" card (System Settings > System & Embeddings).
 *
 * Its rows are system_settings orchestrator_llm provider + model, read from the same
 * by-category endpoint SystemSettingsTab loads (that tab passes the card nothing).
 * Save goes through bulk-update, which moves the Auto agents still on the platform
 * default to the new one (api/system_settings.py); Reset saves each row's
 * default_value through that same path, so the Auto agents move then too.
 */

import { useCallback, useEffect, useState } from 'react'
import { toast } from 'sonner'
import {
  BulkUpdateReply,
  SystemSetting,
  bulkUpdateSettings,
  getSettingsByCategory,
} from '@/lib/api/system-settings'

export const AUTO_DEFAULT_CATEGORY = 'orchestrator_llm'
/** The orchestrator_llm keys the card shows: the route. */
export const AUTO_DEFAULT_KEYS = ['provider', 'model']

export interface AutoPlatformDefault {
  settings: SystemSetting[]
  loading: boolean
  saving: boolean
  save: (updates: Record<string, string>) => Promise<void>
  reset: () => Promise<void>
}

async function loadAutoDefaultSettings(): Promise<SystemSetting[]> {
  const categories = await getSettingsByCategory()
  const rows = categories.find((c) => c.category === AUTO_DEFAULT_CATEGORY)?.settings ?? []
  return rows.filter((s) => AUTO_DEFAULT_KEYS.includes(s.key))
}

/** The success toast: with the Auto agents the backend moved, when the reply counts them. */
function savedMessage(reply: BulkUpdateReply): string {
  const moved = reply.auto_models?.moved
  if (moved === undefined) return 'Platform default for Auto saved'
  return `Platform default for Auto saved; ${moved} Auto agent${moved === 1 ? '' : 's'} moved to it`
}

/** The rows a save sends: each shown setting the form names, by id. */
function bulkUpdates(settings: SystemSetting[], updates: Record<string, string>) {
  return settings
    .filter((s) => updates[s.key] !== undefined)
    .map((s) => ({ id: s.id, value: updates[s.key] }))
}

export function useAutoPlatformDefault(): AutoPlatformDefault {
  const [settings, setSettings] = useState<SystemSetting[]>([])
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)

  const load = useCallback(async () => {
    try {
      setSettings(await loadAutoDefaultSettings())
    } catch (err) {
      console.error('Failed to load the platform default for Auto:', err)
      toast.error('Failed to load the platform default for Auto')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void load()
  }, [load])

  const save = useCallback(async (updates: Record<string, string>) => {
    const changes = bulkUpdates(settings, updates)
    if (changes.length === 0) return
    try {
      setSaving(true)
      toast.success(savedMessage(await bulkUpdateSettings(changes)))
      await load()
    } catch (err) {
      console.error('Failed to save the platform default for Auto:', err)
      toast.error(err instanceof Error ? err.message : 'Failed to save the platform default for Auto')
    } finally {
      setSaving(false)
    }
  }, [settings, load])

  const reset = useCallback(
    () => save(Object.fromEntries(settings.map((s) => [s.key, s.default_value ?? '']))),
    [settings, save],
  )

  return { settings, loading, saving, save, reset }
}
