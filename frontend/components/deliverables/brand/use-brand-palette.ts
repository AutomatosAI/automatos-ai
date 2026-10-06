'use client'

/**
 * PRD-255 US-007 — the colour roles as the Brand kit page edits them. A role the owner
 * changes is `set` (Save sends it); "Reset to derived" stores the role empty
 * (PUT palette {role: ""}, merged key by key on the server) and reads the kit back, so the
 * role shows the colour it now derives from the kit's four colours. A 422 from the save's
 * contrast check names a role (loc ['palette', role]): its message shows under that swatch.
 * A reset is stored at once, so `onStored` runs after it (the brand board redraws, US-010).
 */
import { useCallback, useState, type Dispatch, type SetStateAction } from 'react'
import { toast } from 'sonner'

import { templateBlocksApi } from '@/components/documents/blocks/api'
import type { BrandKit, BrandPaletteRole } from '@/components/documents/blocks/types'
import { PALETTE_ROLES, roleErrorsFrom, saveErrorMessage, type RoleErrors } from './save-errors'

/** ``kit`` with ``role`` set to ``hex``: Save sends it. */
export function withRole(kit: BrandKit, role: BrandPaletteRole, hex: string): BrandKit {
  return {
    ...kit,
    palette: { ...kit.palette, [role]: hex },
    palette_source: { ...kit.palette_source, [role]: 'set' },
  }
}

/**
 * ``local`` with the roles of ``fresh`` (the kit read back after ``reset`` went back to
 * derived), keeping each other role the owner set and has not saved yet.
 */
export function withPaletteFrom(local: BrandKit, fresh: BrandKit, reset: BrandPaletteRole): BrandKit {
  const keep = PALETTE_ROLES.filter((role) => role !== reset && local.palette_source?.[role] === 'set')
  return {
    ...local,
    palette: { ...fresh.palette, ...Object.fromEntries(keep.map((role) => [role, local.palette?.[role]])) },
    palette_source: { ...fresh.palette_source, ...Object.fromEntries(keep.map((role) => [role, 'set' as const])) },
  }
}

function without(errors: RoleErrors, role: BrandPaletteRole): RoleErrors {
  return Object.fromEntries(Object.entries(errors).filter(([key]) => key !== role))
}

export function useBrandPalette(
  setKit: Dispatch<SetStateAction<BrandKit | null>>,
  onStored: () => void,
  // F372: a role changed on the page is an unsaved edit until Save.
  onEdit: () => void,
) {
  const [roleErrors, setRoleErrors] = useState<RoleErrors>({})
  const [resetting, setResetting] = useState<BrandPaletteRole | null>(null)

  const setRole = useCallback((role: BrandPaletteRole, hex: string) => {
    setKit((k) => (k ? withRole(k, role, hex) : k))
    setRoleErrors((errors) => without(errors, role))
    onEdit()
  }, [setKit, onEdit])

  const resetRole = async (role: BrandPaletteRole) => {
    setResetting(role)
    try {
      await templateBlocksApi.updateBrandKit({ palette: { [role]: '' } })
      const fresh = await templateBlocksApi.getBrandKit()
      setKit((k) => (k ? withPaletteFrom(k, fresh, role) : k))
      setRoleErrors((errors) => without(errors, role))
      onStored()
      toast.success(`${role} follows the kit's colours again`)
    } catch (e: any) {
      setRoleErrors(roleErrorsFrom(e))
      toast.error(saveErrorMessage(e))
    } finally {
      setResetting(null)
    }
  }

  return { roleErrors, setRoleErrors, resetting, setRole, resetRole }
}

export type BrandPaletteEdits = ReturnType<typeof useBrandPalette>
