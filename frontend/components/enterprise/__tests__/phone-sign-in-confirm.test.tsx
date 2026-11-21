/**
 * PRD-277 US-603 — the page a phone's sign-in waits on (app/sign-in/phone).
 *
 *   - It reads what waits through apiClient and says who is signing in, to which app, on which phone, and to
 *     continue only if they started this from the app; a phone someone else removed says so.
 *   - Two buttons. Continue posts once and opens the app link the API answered, only when it is the app's own
 *     link; Cancel posts and says the window can be closed.
 *   - Each refusal the API sends here (`?error=`) is its one sentence and no button; an unknown code is the
 *     general sentence, never the code. Nothing waiting is one sentence and no button, at load or at Continue.
 *     A refused Continue (the device cap) shows its sentence.
 *   - In the enterprise build app/sign-in/phone renders this page; elsewhere it is not found.
 */
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const nav = vi.hoisted(() => ({ params: new URLSearchParams(), notFound: vi.fn() }))
const api = vi.hoisted(() => ({ get: vi.fn(), post: vi.fn() }))
const build = vi.hoisted(() => ({ enterprise: true }))
const assign = vi.fn()

vi.mock('next/navigation', () => ({ useSearchParams: () => nav.params, notFound: nav.notFound }))
vi.mock('@/lib/api-client', () => ({ apiClient: { get: api.get, post: api.post, getBaseUrl: () => '' } }))
vi.mock('@/lib/edition-name', () => ({
  get isEnterprise() {
    return build.enterprise
  },
  resolveEdition: () => (build.enterprise ? 'enterprise' : 'saas'),
}))

import PhoneSignInPage from '@/app/sign-in/phone/page'
import { PhoneSignInConfirm } from '@/components/enterprise/phone-sign-in-confirm'
import {
  PHONE_CANCELLED,
  PHONE_CANCEL_PATH,
  PHONE_CONTINUE_PATH,
  PHONE_FAILED,
  PHONE_NOTHING_WAITING,
  PHONE_ONLY_IF_YOU_STARTED,
  PHONE_REFUSALS,
  PHONE_REMOVED_BEFORE,
  PHONE_WAITING_PATH,
} from '@/lib/enterprise/phone-sign-in'

const LINK = 'automatos://go/signed-in?code=c0de'
const WAITING = { person: 'sam@example.test', app: 'Automatos Go', platform: 'ios', removed_before: false }

function refused(status: number, code: string) {
  return Object.assign(new Error(code), { status, body: { detail: { code } } })
}

beforeEach(() => {
  vi.stubGlobal('location', { ...window.location, assign })
  vi.spyOn(console, 'warn').mockImplementation(() => {})
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
  nav.params = new URLSearchParams()
  api.get.mockReset()
  api.post.mockReset()
  assign.mockReset()
})

async function shown(waiting: object = WAITING) {
  api.get.mockResolvedValue(waiting)
  render(<PhoneSignInConfirm />)
  const button = await screen.findByRole('button', { name: 'Continue' })
  await waitFor(() => expect(button).toBeEnabled())
  return button
}

describe('PRD-277 US-603 — the phone sign-in page', () => {
  it('says who signs in to which app on which phone, and to continue only if they started it', async () => {
    await shown()
    expect(api.get).toHaveBeenCalledWith(PHONE_WAITING_PATH)
    const said = 'You are signing in to the Automatos Go app on this iPhone as sam@example.test.'
    expect(screen.getByText(said)).toBeTruthy()
    expect(screen.getByText(PHONE_ONLY_IF_YOU_STARTED)).toBeTruthy()
    expect(screen.queryByText(PHONE_REMOVED_BEFORE)).toBeNull()
    expect(screen.getAllByRole('button').map((button) => button.textContent)).toEqual(['Continue', 'Cancel'])
  })

  it('says when someone else removed this phone before', async () => {
    await shown({ ...WAITING, platform: 'android', removed_before: true })
    expect(screen.getByText(PHONE_REMOVED_BEFORE)).toBeTruthy()
    expect(screen.getByText(/on this Android phone as sam@example.test/)).toBeTruthy()
  })

  it('Continue posts once and opens the app link the API answered', async () => {
    api.post.mockResolvedValue({ app_link: LINK })
    fireEvent.click(await shown())

    await waitFor(() => expect(assign).toHaveBeenCalledWith(LINK))
    expect(api.post).toHaveBeenCalledTimes(1)
    expect(api.post).toHaveBeenCalledWith(PHONE_CONTINUE_PATH)
    expect(screen.queryAllByRole('button')).toHaveLength(0)
  })

  it('opens nothing but the app’s own link', async () => {
    api.post.mockResolvedValue({ app_link: 'https://attacker.example/catch?code=c0de' })
    fireEvent.click(await shown())

    expect(await screen.findByRole('alert')).toHaveTextContent(PHONE_FAILED)
    expect(assign).not.toHaveBeenCalled()
  })

  it('Cancel posts and says the window can be closed', async () => {
    api.post.mockResolvedValue(null)
    await shown()
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))

    expect(await screen.findByText(PHONE_CANCELLED)).toBeTruthy()
    expect(api.post).toHaveBeenCalledWith(PHONE_CANCEL_PATH)
    expect(screen.queryAllByRole('button')).toHaveLength(0)
    expect(assign).not.toHaveBeenCalled()
  })

  it('a Continue refused at the device cap shows its sentence', async () => {
    api.post.mockRejectedValue(refused(409, 'device_limit_reached'))
    fireEvent.click(await shown())

    expect(await screen.findByRole('alert')).toHaveTextContent(PHONE_REFUSALS.device_limit_reached)
    expect(screen.queryAllByRole('button')).toHaveLength(0)
  })

  it('a Continue that finds nothing waiting says so', async () => {
    api.post.mockRejectedValue(refused(404, 'nothing_waiting'))
    fireEvent.click(await shown())

    expect(await screen.findByText(PHONE_NOTHING_WAITING)).toBeTruthy()
    expect(screen.queryAllByRole('button')).toHaveLength(0)
  })

  it('nothing waiting at load is one sentence and offers nothing', async () => {
    api.get.mockRejectedValue(refused(404, 'nothing_waiting'))
    render(<PhoneSignInConfirm />)

    expect(await screen.findByText(PHONE_NOTHING_WAITING)).toBeTruthy()
    expect(screen.queryAllByRole('button')).toHaveLength(0)
  })

  it.each(Object.keys(PHONE_REFUSALS))('shows the sentence for ?error=%s and asks the API nothing', async (code) => {
    nav.params = new URLSearchParams({ error: code })
    render(<PhoneSignInConfirm />)

    expect(await screen.findByRole('alert')).toHaveTextContent(PHONE_REFUSALS[code])
    expect(screen.queryAllByRole('button')).toHaveLength(0)
    expect(api.get).not.toHaveBeenCalled()
  })

  it('shows the general sentence for a code it does not know, never the code', async () => {
    nav.params = new URLSearchParams({ error: '<b>hello</b>' })
    render(<PhoneSignInConfirm />)

    expect(await screen.findByRole('alert')).toHaveTextContent(PHONE_FAILED)
    expect(screen.queryByText(/hello/)).toBeNull()
  })
})

describe('app/sign-in/phone', () => {
  afterEach(() => {
    build.enterprise = true
    nav.notFound.mockReset()
  })

  it('renders the page in the enterprise build', async () => {
    api.get.mockResolvedValue(WAITING)
    render(<PhoneSignInPage />)

    expect(await screen.findByRole('heading', { name: 'Sign in on your phone' })).toBeTruthy()
    expect(nav.notFound).not.toHaveBeenCalled()
  })

  it('is not found in the other builds', () => {
    build.enterprise = false
    nav.notFound.mockImplementation(() => {
      throw new Error('NEXT_NOT_FOUND')
    })

    // The page calls no hook before notFound(), so it is called as the function Next calls.
    expect(() => PhoneSignInPage()).toThrow('NEXT_NOT_FOUND')
    expect(nav.notFound).toHaveBeenCalledTimes(1)
    expect(api.get).not.toHaveBeenCalled()
  })
})
