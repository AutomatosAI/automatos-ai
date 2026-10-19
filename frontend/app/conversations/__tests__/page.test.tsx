/**
 * PRD-260 T-006, K10 — /conversations is Play's alone: the server component
 * reads the automatos-style cookie (parseUiStyle) and sends Classic, Studio, a
 * missing and a tampered cookie to /chat before anything paints; under Play it
 * renders MainLayout (fullBleed) with Play's Conversations screen.
 *
 * PRD-261 A-02b: "under Play" is the style builtin Play's, `theme:builtin:play`,
 * which the legacy `play` cookie parses to as well; any other theme (an installed
 * one) has no Conversations page here and goes to /chat.
 */
import React from 'react'
import { cleanup, render, screen } from '@testing-library/react'
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

vi.mock('@/components/layout/main-layout', () => ({
  MainLayout: ({ children, fullBleed }: { children: React.ReactNode; fullBleed?: boolean }) => (
    <div data-testid="main-layout" data-full-bleed={fullBleed ? 'true' : 'false'}>
      {children}
    </div>
  ),
}))
vi.mock('@/components/play/conversations/play-conversations', () => ({
  PlayConversations: () => <div data-testid="play-conversations" />,
}))

import ConversationsPage from '../page'

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  cookieJar.style = undefined
})

/** An installed theme's style: a lowercase uuid after `theme:install:`. */
const INSTALLED_STYLE = 'theme:install:0b7c6a52-3f1d-4e8a-9c2b-5d4e6f7a8b9c'

describe('/conversations', () => {
  it.each([
    ['the style builtin Play is', 'theme:builtin:play'],
    ['a legacy play cookie', 'play'],
  ])('renders Play’s Conversations inside MainLayout under %s', async (_label, value) => {
    cookieJar.style = value
    render(await ConversationsPage())
    expect(redirectMock).not.toHaveBeenCalled()
    expect(screen.getByTestId('main-layout')).toHaveAttribute('data-full-bleed', 'true')
    expect(screen.getByTestId('play-conversations')).toBeInTheDocument()
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
