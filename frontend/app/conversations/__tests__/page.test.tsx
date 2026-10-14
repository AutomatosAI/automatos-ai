/**
 * PRD-260 T-006, K10 — /conversations is Play's alone: the server component
 * reads the automatos-style cookie (parseUiStyle) and sends Classic, Studio, a
 * missing and a tampered cookie to /chat before anything paints; under Play it
 * renders MainLayout (fullBleed) with Play's Conversations screen.
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

describe('/conversations', () => {
  it('renders Play’s Conversations inside MainLayout under Play', async () => {
    cookieJar.style = 'play'
    render(await ConversationsPage())
    expect(redirectMock).not.toHaveBeenCalled()
    expect(screen.getByTestId('main-layout')).toHaveAttribute('data-full-bleed', 'true')
    expect(screen.getByTestId('play-conversations')).toBeInTheDocument()
  })

  it.each([
    ['Studio', 'studio'],
    ['Classic', 'classic'],
    ['a tampered cookie (the default style)', 'play; studio'],
    ['no cookie (the default style)', undefined],
  ])('redirects to /chat under %s', async (_label, value) => {
    cookieJar.style = value
    await expect(ConversationsPage()).rejects.toThrow('NEXT_REDIRECT:/chat')
    expect(redirectMock).toHaveBeenCalledWith('/chat')
  })
})
