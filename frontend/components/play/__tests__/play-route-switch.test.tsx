/**
 * PRD-260 K7, D7, D8 — the switch Play's pages mount through, and the chat
 * layout that uses it: Play's chat under Play at >= 1024 px on /chat, the
 * existing page under Classic and Studio, below 1024 px, on another path, and
 * for a link into a surface only the core chat has (the Code canvas, a
 * session's runtime, plan mode). The DiscussionBar stays either way.
 *
 * P260-FX-11 (M13, M14): a value Play's screen serves (`playServes`) keeps
 * Play, and the switch carries its own Suspense boundary whose fallback is the
 * route's own page, so a search-params read that suspends (a static
 * prerender) renders the page.
 * P260-FX-23 (M17): the layout hands the switch Play's chat through its lazy
 * slot, so the chat arrives a moment after the first render (findBy*).
 * P260-FX-25 (M69): the width comes from useThemeLayout (stood in for here;
 * play-layout-phone.test.tsx runs the real one): while it isn't known (the
 * server, hydration) the switch renders neither Play's screen nor the page
 * where the width decides, and the page where it doesn't.
 */
import React from 'react'
import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

const state = vi.hoisted(() => ({
  play: true,
  narrow: false,
  /** The width isn't known yet (the server, hydration): useThemeLayout is 'pending' under Play. */
  widthUnknown: false,
  pathname: '/chat',
  search: '',
  /** While set, reading the search params suspends on it (Next's static-prerender bailout). */
  pending: null as Promise<void> | null,
}))

vi.mock('next/navigation', () => ({
  usePathname: () => state.pathname,
  useSearchParams: () => {
    if (state.pending !== null) throw state.pending
    return new URLSearchParams(state.search)
  },
}))
vi.mock('@/hooks/use-studio-theme', async () => {
  const { BUILTIN_PLAY_ENTRY } = await import('@/lib/theme-host/builtin/play-entry')
  return {
    useIsStudio: () => true,
    useActiveTheme: () => (state.play ? BUILTIN_PLAY_ENTRY : null),
    useThemeLayout: () => {
      if (!state.play || state.narrow) return 'fallback'
      return state.widthUnknown ? 'pending' : 'theme'
    },
  }
})
vi.mock('@/components/play/chat/play-chat', () => ({ PlayChat: () => <div data-testid="play-chat">Play chat</div> }))
vi.mock('@/components/chatbot/discussion-bar', () => ({ DiscussionBar: () => <div data-testid="discussion-bar" /> }))

import ChatLayout from '@/app/chat/layout'
import { PlayRouteSwitch } from '@/components/play/play-route-switch'

function renderSwitch(props: { paths?: string[]; keepPageFor?: string[]; playServes?: Record<string, string[]> } = {}) {
  return render(
    <PlayRouteSwitch
      play={<div data-testid="play">Play</div>}
      paths={props.paths ?? ['/chat']}
      keepPageFor={props.keepPageFor}
      playServes={props.playServes}
    >
      <div data-testid="page">The page</div>
    </PlayRouteSwitch>,
  )
}

function shows(testId: 'play' | 'page') {
  expect(screen.getByTestId(testId)).toBeInTheDocument()
  expect(screen.queryByTestId(testId === 'play' ? 'page' : 'play')).toBeNull()
}

afterEach(() => {
  Object.assign(state, { play: true, narrow: false, widthUnknown: false, pathname: '/chat', search: '', pending: null })
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

  it('renders neither Play nor the page while the width is not known, so neither mounts for a render it then loses (M69)', () => {
    state.widthUnknown = true
    renderSwitch()
    expect(screen.queryByTestId('play')).toBeNull()
    expect(screen.queryByTestId('page')).toBeNull()
  })

  it('renders the page while the width is not known where the width does not decide (another path, a page asked for)', () => {
    state.widthUnknown = true
    state.pathname = '/chat/elsewhere'
    renderSwitch()
    shows('page')
    cleanup()
    state.pathname = '/chat'
    state.search = 'repo=projects%2Fsite'
    renderSwitch({ keepPageFor: ['repo'] })
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

  describe('playServes: values of a page parameter that Play still serves', () => {
    const served = { keepPageFor: ['tab', 'post'], playServes: { tab: ['outputs'] } }

    it("keeps Play's screen for a value Play serves", () => {
      state.search = 'tab=outputs'
      renderSwitch(served)
      shows('play')
    })

    it.each(['tab=brand', 'tab=', 'tab=outputs&post=p1'])('renders the page for ?%s', (search) => {
      state.search = search
      renderSwitch(served)
      shows('page')
    })
  })

  it("renders the page as the fallback while the search params suspend (a static prerender), then Play's screen", async () => {
    let release = () => {}
    state.pending = new Promise<void>((resolve) => {
      release = resolve
    })
    renderSwitch()
    shows('page')
    state.pending = null
    await act(async () => {
      release()
    })
    expect(await screen.findByTestId('play')).toBeInTheDocument()
    expect(screen.queryByTestId('page')).toBeNull()
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

  it("renders Play's chat instead of the chat page under Play at desktop width, with the DiscussionBar", async () => {
    renderLayout()
    expect(await screen.findByTestId('play-chat')).toBeInTheDocument()
    expect(screen.queryByTestId('page')).toBeNull()
    expect(screen.getByTestId('discussion-bar')).toBeInTheDocument()
  })

  it('renders neither chat while the width is not known, the DiscussionBar still there (M69)', () => {
    state.widthUnknown = true
    renderLayout()
    expect(screen.queryByTestId('page')).toBeNull()
    expect(screen.queryByTestId('play-chat')).toBeNull()
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
