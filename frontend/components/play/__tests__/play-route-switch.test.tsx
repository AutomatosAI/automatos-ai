/**
 * PRD-260 K7, D7, D8 — the switch Play's pages mount through, and the chat
 * layout that uses it: Play's chat under Play at >= 1024 px on /chat, the
 * existing page under Classic and Studio, below 1024 px, on another path, and
 * for a link into a surface only the core chat has (the Code canvas, a
 * session's runtime, plan mode). The DiscussionBar stays either way.
 */
import React from 'react'
import { render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

const state = vi.hoisted(() => ({ play: true, narrow: false, pathname: '/chat', search: '' }))

vi.mock('next/navigation', () => ({
  usePathname: () => state.pathname,
  useSearchParams: () => new URLSearchParams(state.search),
}))
vi.mock('@/hooks/use-mobile', () => ({ useIsTabletOrBelow: () => state.narrow, useIsMobile: () => state.narrow }))
vi.mock('@/hooks/use-studio-theme', () => ({ useIsStudio: () => true, useIsPlay: () => state.play }))
vi.mock('@/components/play/chat/play-chat', () => ({ PlayChat: () => <div data-testid="play-chat">Play chat</div> }))
vi.mock('@/components/chatbot/discussion-bar', () => ({ DiscussionBar: () => <div data-testid="discussion-bar" /> }))

import ChatLayout from '@/app/chat/layout'
import { PlayRouteSwitch } from '@/components/play/play-route-switch'

function renderSwitch(props: { paths?: string[]; keepPageFor?: string[] } = {}) {
  return render(
    <PlayRouteSwitch play={<div data-testid="play">Play</div>} paths={props.paths ?? ['/chat']} keepPageFor={props.keepPageFor}>
      <div data-testid="page">The page</div>
    </PlayRouteSwitch>,
  )
}

function shows(testId: 'play' | 'page') {
  expect(screen.getByTestId(testId)).toBeInTheDocument()
  expect(screen.queryByTestId(testId === 'play' ? 'page' : 'play')).toBeNull()
}

afterEach(() => {
  Object.assign(state, { play: true, narrow: false, pathname: '/chat', search: '' })
})

describe('PlayRouteSwitch', () => {
  it("renders Play's screen under Play at desktop width on one of its paths", () => {
    renderSwitch()
    shows('play')
  })

  it('renders the page under Classic or Studio', () => {
    state.play = false
    renderSwitch()
    shows('page')
  })

  it('renders the page below 1024 px, as Studio falls back (D7)', () => {
    state.narrow = true
    renderSwitch()
    shows('page')
  })

  it('renders the page on a path Play has no screen for', () => {
    state.pathname = '/chat/elsewhere'
    renderSwitch()
    shows('page')
  })

  it('renders the page when the link asks for a surface only the page has', () => {
    state.search = 'repo=projects%2Fsite&ticket=42'
    renderSwitch({ keepPageFor: ['repo', 'runtime', 'mode'] })
    shows('page')
  })

  it('keeps Play for any other query (a deep link to a conversation)', () => {
    state.search = 'chatId=c1'
    renderSwitch({ keepPageFor: ['repo', 'runtime', 'mode'] })
    shows('play')
  })
})

describe('app/chat/layout.tsx', () => {
  function renderLayout() {
    return render(
      <ChatLayout>
        <div data-testid="page">The chat page</div>
      </ChatLayout>,
    )
  }

  it("renders Play's chat instead of the chat page under Play at desktop width, with the DiscussionBar", () => {
    renderLayout()
    expect(screen.getByTestId('play-chat')).toBeInTheDocument()
    expect(screen.queryByTestId('page')).toBeNull()
    expect(screen.getByTestId('discussion-bar')).toBeInTheDocument()
  })

  it('renders the existing chat page otherwise', () => {
    state.play = false
    renderLayout()
    expect(screen.getByTestId('page')).toHaveTextContent('The chat page')
    expect(screen.queryByTestId('play-chat')).toBeNull()
    expect(screen.getByTestId('discussion-bar')).toBeInTheDocument()
  })

  it.each(['repo=projects%2Fsite', 'ticket=42&repo=x&runtime=1', 'mode=plan&from=assignments'])(
    'keeps the existing chat page for ?%s under Play',
    (search) => {
      state.search = search
      renderLayout()
      expect(screen.getByTestId('page')).toBeInTheDocument()
      expect(screen.queryByTestId('play-chat')).toBeNull()
    },
  )
})
