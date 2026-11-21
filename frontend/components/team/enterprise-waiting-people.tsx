'use client'

/**
 * PRD-277 US-504 — the Team page's "Waiting for a workspace" card (enterprise, install admins):
 * each person who signed in and holds no workspace, oldest first, when they first signed in, a role
 * to give them and "Add to workspace"; a page at a time, with Previous and Next. Drawn like the
 * members card (members-table.tsx, member-row.tsx).
 */
import { useState } from 'react'
import { Mail, UserCheck } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { formatAbsoluteDate } from '@/lib/format-relative'
import { PLACEABLE_ROLES, type PlaceableRole, type WaitingPerson } from '@/lib/enterprise/auth-api'
import { NEXT_PAGE, PREVIOUS_PAGE, WAITING_PAGE_LABEL } from '@/lib/enterprise/messages'

import type { WaitingPeopleState } from './use-enterprise-waiting-people'

type OnPlace = (userId: number, role: PlaceableRole) => void

const ROLE_LABELS: Readonly<Record<PlaceableRole, string>> = { viewer: 'Viewer', editor: 'Editor', admin: 'Admin' }

function WaitingRow({ person, onPlace }: { person: WaitingPerson; onPlace: OnPlace }) {
  const [role, setRole] = useState<PlaceableRole>('viewer')
  const label = person.name || person.email || `Person ${person.user_id}`

  return (
    <div className="flex flex-col md:grid md:grid-cols-12 gap-2 md:gap-4 p-4 items-start md:items-center">
      <div className="md:col-span-5 min-w-0">
        <div className="font-medium text-foreground truncate">{label}</div>
        {person.email && (
          <div className="text-sm text-muted-foreground flex items-center gap-1 truncate">
            <Mail className="w-3 h-3 shrink-0" />
            {person.email}
          </div>
        )}
      </div>
      <div className="md:col-span-3 text-sm text-muted-foreground">
        {person.signed_in_first_at ? `First signed in ${formatAbsoluteDate(person.signed_in_first_at)}` : '—'}
      </div>
      <div className="md:col-span-2">
        <label className="sr-only" htmlFor={`waiting-role-${person.user_id}`}>Role for {label}</label>
        <select
          id={`waiting-role-${person.user_id}`}
          value={role}
          onChange={(event) => setRole(event.target.value as PlaceableRole)}
          className="w-full px-3 py-1.5 text-sm rounded-lg border border-border bg-transparent focus:outline-none focus:ring-1 focus:ring-primary"
        >
          {PLACEABLE_ROLES.map((value) => (
            <option key={value} value={value} className="bg-popover text-popover-foreground">{ROLE_LABELS[value]}</option>
          ))}
        </select>
      </div>
      <div className="md:col-span-2 md:text-right">
        <Button variant="outline" size="sm" onClick={() => onPlace(person.user_id, role)}>
          <UserCheck className="w-4 h-4 mr-2" />
          Add to workspace
        </Button>
      </div>
    </div>
  )
}

type PagerProps = Pick<WaitingPeopleState, 'total' | 'offset' | 'pageSize' | 'loading' | 'goTo'> & { readonly shown: number }

function WaitingPager({ total, offset, pageSize, loading, goTo, shown }: PagerProps) {
  if (total <= pageSize && offset === 0) return null
  return (
    <div className="flex items-center justify-between gap-2 p-4 border-t border-border/30 text-sm text-muted-foreground">
      <span>{WAITING_PAGE_LABEL(offset + 1, offset + shown, total)}</span>
      <div className="flex gap-2">
        <Button variant="outline" size="sm" disabled={loading || offset === 0} onClick={() => goTo(offset - pageSize)}>
          {PREVIOUS_PAGE}
        </Button>
        <Button variant="outline" size="sm" disabled={loading || offset + shown >= total} onClick={() => goTo(offset + pageSize)}>
          {NEXT_PAGE}
        </Button>
      </div>
    </div>
  )
}

export function EnterpriseWaitingPeople({ waiting }: { readonly waiting: WaitingPeopleState }) {
  if (waiting.total === 0 && waiting.people.length === 0) return null
  const onPlace: OnPlace = (userId, role) => void waiting.handlePlace(userId, role)
  return (
    <section aria-labelledby="waiting-people-title" className="glass-card rounded-xl border border-border/30 overflow-hidden">
      <div className="p-4 border-b border-border/30 bg-white/5">
        <h2 id="waiting-people-title" className="font-medium text-foreground">Waiting for a workspace</h2>
        <p className="text-sm text-muted-foreground">
          They signed in with your organisation&apos;s identity provider and are not in a workspace yet.
        </p>
      </div>
      <div className="divide-y divide-white/5">
        {waiting.people.map((person) => (
          <WaitingRow key={person.user_id} person={person} onPlace={onPlace} />
        ))}
      </div>
      <WaitingPager {...waiting} shown={waiting.people.length} />
    </section>
  )
}
