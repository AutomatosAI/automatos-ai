'use client'

/**
 * PRD-277 US-504 — people waiting for a workspace, for an install admin in enterprise: they signed
 * in at the organisation's identity provider and hold no workspace yet. Listed a page at a time,
 * oldest first, from `GET /api/workspaces/{id}/team/waiting?limit=&offset=` (the API states the
 * total); placing one in this workspace is `POST …/waiting/{user_id}` with a role below owner,
 * which asks for a recent sign-in (step-up: the browser signs in again, and the admin places the
 * person once more on return).
 *
 * Only EnterpriseWaitingSection calls this, and only enterprise renders that, so `local` and
 * `saas` run no hook of it. The API decides who may see or do this; `enabled` only spares a
 * request it would refuse.
 */
import { useCallback, useEffect, useState } from 'react'

import {
  WAITING_PAGE_SIZE,
  fetchWaitingPeople,
  placeWaitingPerson,
  type PlaceableRole,
  type WaitingPerson,
} from '@/lib/enterprise/auth-api'
import { actionErrorMessage } from '@/lib/enterprise/messages'
import { useEnterpriseStepUp } from '@/lib/enterprise/step-up'

export interface WaitingPeopleState {
  readonly people: readonly WaitingPerson[]
  readonly total: number
  readonly offset: number
  readonly pageSize: number
  readonly loading: boolean
  readonly goTo: (offset: number) => void
  readonly handlePlace: (userId: number, role: PlaceableRole) => Promise<void>
}

export function useEnterpriseWaitingPeople(workspaceId: string | null, enabled: boolean, onPlaced: () => void): WaitingPeopleState {
  const [page, setPage] = useState({ people: [] as readonly WaitingPerson[], total: 0, offset: 0 })
  const [loading, setLoading] = useState(false)
  const place = useEnterpriseStepUp(placeWaitingPerson)

  const load = useCallback(async (offset: number) => {
    if (!enabled || !workspaceId) return
    setLoading(true)
    try {
      const answer = await fetchWaitingPeople(workspaceId, offset)
      setPage({ people: answer.people, total: answer.total, offset: answer.offset })
    } catch (error) {
      console.error('[team] the people waiting could not be listed', error)
    } finally {
      setLoading(false)
    }
  }, [enabled, workspaceId])

  useEffect(() => {
    void load(0)
  }, [load])

  const handlePlace = async (userId: number, role: PlaceableRole) => {
    if (!workspaceId) return
    try {
      await place(workspaceId, userId, role)
      // The page they were on, unless it is now past the end.
      const last = Math.max(0, page.total - 2)
      await load(Math.min(page.offset, last - (last % WAITING_PAGE_SIZE)))
      onPlaced()
    } catch (error) {
      console.error('[team] the person could not be added', error)
      alert(actionErrorMessage(error))
    }
  }

  return {
    ...page,
    pageSize: WAITING_PAGE_SIZE,
    loading,
    goTo: (offset: number) => void load(Math.max(0, offset)),
    handlePlace,
  }
}
