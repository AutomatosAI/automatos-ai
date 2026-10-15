/**
 * PRD-260 F5 (Gerard, 15 Oct) — looks, the active look and the seen
 * Deliverables record are kept per signed-in user and workspace in this
 * browser. AccountLookScope, as components/providers.tsx mounts it, gives the
 * LookProvider and the shelf (useCelebrateNew) the account's scope once Clerk
 * and the workspace are known. A second account in the same browser sees
 * neither the first account's looks nor its seen record; the local edition's
 * one user (no user id) keeps its looks; what a browser kept before F5 moves
 * to the first account once and is never deleted.
 */
import { act, cleanup, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const account = vi.hoisted(() => ({
  isLoaded: true,
  isSignedIn: true,
  userId: 'user_a' as string | null,
  workspaceId: 'ws-1' as string | null,
  workspaceLoading: false,
}))

vi.mock('@/lib/auth-hooks', () => ({
  useUser: () => ({
    isLoaded: account.isLoaded,
    isSignedIn: account.isSignedIn,
    user: account.userId === null ? null : { id: account.userId },
  }),
}))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspaceOptional: () => ({
    workspace: account.workspaceId === null ? null : { id: account.workspaceId },
    isLoading: account.workspaceLoading,
  }),
}))

import { AccountLookScope } from '@/components/looks/account-look-scope'
import { LookProvider, useLook } from '@/components/looks/look-provider'
import { UiStyleProvider } from '@/contexts/ui-style-context'
import { useCelebrateNew } from '@/hooks/use-celebrate-new'
import type { Deliverable } from '@/hooks/use-deliverables-api'
import { clearRoot, myLook } from '@/lib/looks/__tests__/helpers'
import { DAYLIGHT } from '@/lib/looks/presets'
import { lookScope } from '@/lib/looks/scope'
import {
  ACTIVE_LOOK_KEY,
  LOOK_SCOPE_KEY,
  SAVED_LOOKS_KEY,
  UNSCOPED_LOOKS_MOVED_KEY,
  lookKeys,
} from '@/lib/looks/storage'
import { PLAY_TONE_COOKIE } from '@/lib/looks/tone'
import { SEEN_KEY, seenKey } from '@/lib/play/seen'
import { APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION } from '@/lib/ui-style'

const A = lookScope('ws-1', 'user_a')
const B = lookScope('ws-1', 'user_b')

function deliverable(id: string, createdAt: string): Deliverable {
  return { id, created_at: createdAt } as Deliverable
}

const OLD = deliverable('d-old', '2026-10-10T09:00:00+00:00')
const NEW = deliverable('d-new', '2026-10-14T09:00:00+00:00')
const OLD_ONLY: readonly Deliverable[] = [OLD]
const BOTH: readonly Deliverable[] = [NEW, OLD]
const OLD_SEEN = { ids: ['d-old'], newest: OLD.created_at }

const celebrate = vi.fn()

function Probe() {
  const { look, saved, loaded, setLook, saveLook } = useLook()
  return (
    <div>
      <p>Loaded: {loaded ? 'yes' : 'no'}</p>
      <p>Active: {look.name}</p>
      <p>Saved: {saved.map((entry) => entry.name).join(', ') || 'none'}</p>
      <button type="button" onClick={() => saveLook(myLook())}>
        Save mine
      </button>
      <button type="button" onClick={() => setLook(DAYLIGHT)}>
        Use Daylight
      </button>
    </div>
  )
}

/** Play's shelf as far as the seen record goes: Auto celebrates a Deliverable this account hasn't seen. */
function Shelf({ items }: { readonly items: readonly Deliverable[] }) {
  const latest = useCelebrateNew(items, celebrate)
  return <p>Celebrating: {latest?.id ?? 'nothing'}</p>
}

/** The providers as components/providers.tsx nests them: the account's scope above the LookProvider. */
function App({ items }: { readonly items: readonly Deliverable[] }) {
  return (
    <UiStyleProvider initialStyle="play">
      <AccountLookScope>
        <LookProvider>
          <Probe />
          <Shelf items={items} />
        </LookProvider>
      </AccountLookScope>
    </UiStyleProvider>
  )
}

function click(name: string): void {
  act(() => screen.getByRole('button', { name }).click())
}

function stored(key: string): unknown {
  const raw = window.localStorage.getItem(key)
  return raw === null ? null : JSON.parse(raw)
}

function signInAs(userId: string | null, workspaceId: string | null = 'ws-1'): void {
  account.userId = userId
  account.workspaceId = workspaceId
}

beforeEach(() => {
  Object.assign(account, { isLoaded: true, isSignedIn: true, userId: 'user_a', workspaceId: 'ws-1', workspaceLoading: false })
  window.localStorage.clear()
  window.localStorage.setItem(APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION)
  document.cookie = `${PLAY_TONE_COOKIE}=; Max-Age=0; Path=/`
  clearRoot()
  celebrate.mockClear()
  vi.spyOn(console, 'warn').mockImplementation(() => undefined)
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  window.localStorage.clear()
  clearRoot()
})

describe('a second account in the same browser (F5)', () => {
  it("sees neither the first account's looks nor its seen Deliverables record", async () => {
    const { rerender } = render(<App items={OLD_ONLY} />)
    expect(await screen.findByText('Loaded: yes')).toBeInTheDocument()
    click('Save mine')
    click('Use Daylight')
    expect(screen.getByText('Saved: My look 1')).toBeInTheDocument()
    expect(stored(seenKey(A))).toEqual(OLD_SEEN)

    signInAs('user_b')
    rerender(<App items={BOTH} />)
    await waitFor(() => expect(screen.getByText('Saved: none')).toBeInTheDocument())
    expect(screen.getByText('Active: Ember')).toBeInTheDocument()
    // Had B read A's record, d-new would be newer than the newest seen: a celebration.
    expect(celebrate).not.toHaveBeenCalled()
    expect(screen.getByText('Celebrating: nothing')).toBeInTheDocument()
    expect(stored(seenKey(B))).toEqual({ ids: ['d-new', 'd-old'], newest: NEW.created_at })

    // B wrote nothing over A's keys.
    expect(stored(lookKeys(A).saved)).toEqual([myLook()])
    expect(stored(lookKeys(A).active)).toEqual(DAYLIGHT)
    expect(stored(seenKey(A))).toEqual(OLD_SEEN)
    expect(stored(lookKeys(B).saved)).toBeNull()
    expect(stored(lookKeys(B).active)).toBeNull()

    signInAs('user_a')
    rerender(<App items={BOTH} />)
    await waitFor(() => expect(screen.getByText('Saved: My look 1')).toBeInTheDocument())
    expect(screen.getByText('Active: Daylight')).toBeInTheDocument()
    await waitFor(() => expect(celebrate).toHaveBeenCalledTimes(1))
    expect(screen.getByText('Celebrating: d-new')).toBeInTheDocument()
  })

  it('keeps the same user apart in another workspace', async () => {
    const { rerender } = render(<App items={OLD_ONLY} />)
    expect(await screen.findByText('Loaded: yes')).toBeInTheDocument()
    click('Save mine')
    signInAs('user_a', 'ws-2')
    rerender(<App items={OLD_ONLY} />)
    await waitFor(() => expect(screen.getByText('Saved: none')).toBeInTheDocument())
    expect(stored(lookKeys(lookScope('ws-2', 'user_a')).saved)).toBeNull()
    expect(stored(lookKeys(A).saved)).toEqual([myLook()])
  })

  it("points the pre-paint script at the account's own active look", async () => {
    const { rerender } = render(<App items={OLD_ONLY} />)
    await waitFor(() => expect(window.localStorage.getItem(LOOK_SCOPE_KEY)).toBe(A))
    signInAs('user_b')
    rerender(<App items={OLD_ONLY} />)
    await waitFor(() => expect(window.localStorage.getItem(LOOK_SCOPE_KEY)).toBe(B))
  })
})

describe('what this browser kept before F5', () => {
  it('moves to the first account once and is never deleted', async () => {
    window.localStorage.setItem(SAVED_LOOKS_KEY, JSON.stringify([myLook()]))
    window.localStorage.setItem(ACTIVE_LOOK_KEY, JSON.stringify(myLook()))
    window.localStorage.setItem(SEEN_KEY, JSON.stringify(OLD_SEEN))
    const { rerender } = render(<App items={BOTH} />)
    expect(await screen.findByText('Saved: My look 1')).toBeInTheDocument()
    expect(screen.getByText('Active: My look 1')).toBeInTheDocument()
    // The record moved with them: d-new is newer than the newest this account had seen.
    await waitFor(() => expect(celebrate).toHaveBeenCalledTimes(1))
    expect(window.localStorage.getItem(UNSCOPED_LOOKS_MOVED_KEY)).toBe(A)

    signInAs('user_b')
    rerender(<App items={BOTH} />)
    await waitFor(() => expect(screen.getByText('Saved: none')).toBeInTheDocument())
    expect(screen.getByText('Active: Ember')).toBeInTheDocument()
    expect(celebrate).toHaveBeenCalledTimes(1)

    expect(stored(SAVED_LOOKS_KEY)).toEqual([myLook()])
    expect(stored(ACTIVE_LOOK_KEY)).toEqual(myLook())
    expect(stored(SEEN_KEY)).toEqual(OLD_SEEN)
  })
})

describe('the local edition', () => {
  it("keeps its one user's looks with no user id", async () => {
    signInAs(null, 'ws-local')
    const { unmount } = render(<App items={OLD_ONLY} />)
    expect(await screen.findByText('Loaded: yes')).toBeInTheDocument()
    click('Save mine')
    const local = lookScope('ws-local', null)
    expect(stored(lookKeys(local).saved)).toEqual([myLook()])
    expect(stored(seenKey(local))).toEqual(OLD_SEEN)
    unmount()

    render(<App items={OLD_ONLY} />)
    expect(await screen.findByText('Saved: My look 1')).toBeInTheDocument()
  })
})

describe('before the account is known', () => {
  it('reads, moves and writes nothing while Clerk or the workspace loads, while signed out, or without a workspace', async () => {
    window.localStorage.setItem(SAVED_LOOKS_KEY, JSON.stringify([myLook()]))
    account.isLoaded = false
    const { rerender } = render(<App items={BOTH} />)
    const waiting: (() => void)[] = [
      () => Object.assign(account, { isLoaded: true, isSignedIn: false }),
      () => Object.assign(account, { isSignedIn: true, workspaceLoading: true }),
      () => Object.assign(account, { workspaceLoading: false, workspaceId: null }),
    ]
    for (const step of [() => undefined, ...waiting]) {
      step()
      rerender(<App items={BOTH} />)
      expect(screen.getByText('Loaded: no')).toBeInTheDocument()
    }
    expect(window.localStorage.getItem(UNSCOPED_LOOKS_MOVED_KEY)).toBeNull()
    expect(window.localStorage.getItem(LOOK_SCOPE_KEY)).toBeNull()
    expect(stored(seenKey(A))).toBeNull()
    expect(celebrate).not.toHaveBeenCalled()

    account.workspaceId = 'ws-1'
    rerender(<App items={BOTH} />)
    expect(await screen.findByText('Saved: My look 1')).toBeInTheDocument()
    expect(screen.getByText('Loaded: yes')).toBeInTheDocument()
  })
})
