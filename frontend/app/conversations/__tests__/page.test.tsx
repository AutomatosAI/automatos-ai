/**
 * PRD-260 T-006, K10; PRD-261 A-04b — /conversations is the old address of Play's Conversations
 * page, which is now builtin Play's theme route at /x/conversations. The server component reads
 * the automatos-style cookie (parseUiStyle) and, before anything paints, sends the browser to
 * /x/conversations under a theme whose registry entry lists the route (builtin Play, the legacy
 * `play` cookie too), and to /chat under Classic, Studio, an installed theme (no entry on the
 * server), and a missing or tampered cookie. app/x/[...path]/page.tsx renders the route
 * (components/theme-host/__tests__/theme-slot-route.test.tsx).
 */
import { afterEach, describe, expect, it, vi } from 'vitest'

const cookieJar = vi.hoisted(() => ({ style: undefined as string | undefined }))
vi.mock('next/headers', () => ({
  cookies: async () => ({
    get: (name: string) =>
      name === 'automatos-style' && cookieJar.style !== undefined ? { name, value: cookieJar.style } : undefined,
  }),
}))

const redirectMock = vi.hoisted(() =>
  vi.fn((to: string) => {
    throw new Error(`NEXT_REDIRECT:${to}`)
  }),
)
vi.mock('next/navigation', () => ({ redirect: redirectMock }))

import ConversationsPage from '../page'

afterEach(() => {
  vi.clearAllMocks()
  cookieJar.style = undefined
})

/** An installed theme's style: a lowercase uuid after `theme:install:`. */
const INSTALLED_STYLE = 'theme:install:0b7c6a52-3f1d-4e8a-9c2b-5d4e6f7a8b9c'

describe('/conversations', () => {
  it.each([
    ['the style builtin Play is', 'theme:builtin:play'],
    ['a legacy play cookie', 'play'],
  ])('sends the browser to /x/conversations under %s', async (_label, value) => {
    cookieJar.style = value
    await expect(ConversationsPage()).rejects.toThrow('NEXT_REDIRECT:/x/conversations')
    expect(redirectMock).toHaveBeenCalledTimes(1)
    expect(redirectMock).toHaveBeenCalledWith('/x/conversations')
  })

  it.each([
    ['Studio', 'studio'],
    ['Classic', 'classic'],
    ['a tampered cookie (the default style)', 'play; studio'],
    ['an installed theme', INSTALLED_STYLE],
    ['no cookie (the default style)', undefined],
  ])('redirects to /chat under %s', async (_label, value) => {
    cookieJar.style = value
    await expect(ConversationsPage()).rejects.toThrow('NEXT_REDIRECT:/chat')
    expect(redirectMock).toHaveBeenCalledWith('/chat')
  })
})
