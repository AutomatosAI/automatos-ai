'use client'

/**
 * PRD-259 §4 — "Add a skin…": a .autoskin goes to POST /api/skins/import through apiClient
 * (FormData), the installed list refreshes and the new skin becomes active. A refused
 * package (400, or 413 over the size cap) shows the validator's messages in a toast.
 */
import { useCallback, useState } from 'react'
import { toast } from 'sonner'
import { useUiStyle } from '@/contexts/ui-style-context'
import { apiClient } from '@/lib/api-client'
import {
  IMPORT_FILE_FIELD,
  IMPORT_SKIN_PATH,
  importErrorMessages,
  parseInstalledSkin,
} from '@/lib/skins/installed'
import { skinStyle } from '@/lib/ui-style'

/** At most this many of the validator's messages in one toast; the rest are counted. */
export const MAX_IMPORT_MESSAGES = 5
export const IMPORT_REFUSED_TITLE = "This skin can't be added."
const IMPORT_FAILED_TITLE = "The skin couldn't be added. Try again."
const HTTP_FORBIDDEN = 403
const IMPORT_FORBIDDEN_TITLE = "You can't add skins to this workspace."
/** One message per line in the toast. */
const MESSAGES_CLASS = 'whitespace-pre-line'

function statusOf(error: unknown): number | null {
  const status = (error as { status?: unknown } | null)?.status
  return typeof status === 'number' ? status : null
}

/** The validator's messages as one toast description, or undefined when there are none. */
export function importErrorDescription(messages: readonly string[]): string | undefined {
  if (messages.length === 0) return undefined
  const shown = messages.slice(0, MAX_IMPORT_MESSAGES)
  const more = messages.length - shown.length
  return more > 0 ? [...shown, `…and ${more} more.`].join('\n') : shown.join('\n')
}

function reportImportError(error: unknown): void {
  console.warn('[Skins] The import was refused.', error)
  const description = importErrorDescription(importErrorMessages(error))
  if (description) toast.error(IMPORT_REFUSED_TITLE, { description, descriptionClassName: MESSAGES_CLASS })
  else toast.error(statusOf(error) === HTTP_FORBIDDEN ? IMPORT_FORBIDDEN_TITLE : IMPORT_FAILED_TITLE)
}

export interface SkinImport {
  readonly importing: boolean
  readonly importFile: (file: File) => Promise<void>
}

export function useSkinImport(): SkinImport {
  const { setStyle, installedSkins } = useUiStyle()
  const { reload } = installedSkins
  const [importing, setImporting] = useState(false)

  const importFile = useCallback(async (file: File) => {
    setImporting(true)
    try {
      const form = new FormData()
      form.append(IMPORT_FILE_FIELD, file)
      const item = parseInstalledSkin(await apiClient.post(IMPORT_SKIN_PATH, form))
      if (item === null) throw new Error('The import answered with something that is not a skin.')
      await reload()
      setStyle(skinStyle(item.id))
    } catch (error) {
      reportImportError(error)
    } finally {
      setImporting(false)
    }
  }, [reload, setStyle])

  return { importing, importFile }
}
