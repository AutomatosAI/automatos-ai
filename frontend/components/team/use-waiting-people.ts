'use client'

/**
 * PRD-277 US-504 — people waiting for a workspace, for an install admin in enterprise: they signed
 * in at the organisation's identity provider and hold no workspace yet. Listed from
 * `GET /api/workspaces/{id}/team/waiting`; placing one in this workspace is
 * `POST …/waiting/{user_id}` with a role below owner, which asks for a recent sign-in (step-up:
 * the browser signs in again, and the admin places the person once more on return).
 *
 * The API decides who may see or do this; `enabled` only spares a request it would refuse.
 */
import { useCallback, useEffect, useState } from 'react'

import { fetchWaitingPeople, placeWaitingPerson, type PlaceableRole, type WaitingPerson } from '@/lib/enterprise/auth-api'
import { themeErrorMessage, useStepUp } from '@/lib/theme-host/api'

export function useWaitingPeople(workspaceId: string | null, enabled: boolean, onPlaced: () => void) {
  const [people, setPeople] = useState<WaitingPerson[]>([])
  const [loading, setLoading] = useState(false)
  const place = useStepUp(placeWaitingPerson)

  const load = useCallback(async () => {
    if (!enabled || !workspaceId) return
    setLoading(true)
    try {
      setPeople((await fetchWaitingPeople(workspaceId)).people)
    } catch (error) {
      console.error('[team] the people waiting could not be listed', error)
    } finally {
      setLoading(false)
    }
  }, [enabled, workspaceId])

  useEffect(() => {
    void load()
  }, [load])

  const handlePlace = async (userId: number, role: PlaceableRole) => {
    if (!workspaceId) return
    try {
      await place(workspaceId, userId, role)
      await load()
      onPlaced()
    } catch (error) {
      console.error('[team] the person could not be added', error)
      alert(themeErrorMessage(error))
    }
  }

  return { people, loading, handlePlace }
}
