/**
 * PRD-260 T-007a, K7, D7, D8 — app/deliverables/layout.tsx mounts Play's
 * shelf through ThemeSlot (PRD-261 A-04b: builtin Play registers it as its
 * /deliverables page, lib/theme-host/builtin/play-slots.tsx): under Play at >= 1024 px on /deliverables
 * only, the existing page under Classic and Studio, below 1024 px, on the
 * explorer, and for a link into one of the page's other sections (a tab, a
 * post, a type drill-down). DeliverableDeepLink is mounted beside the page
 * only: under Play the shelf switch opens a Deliverable as Play's own page
 * (T-007b, K8), so its sheet never opens over it.
 *
 * P260-FX-11 (M13): the Outputs tab IS Play's shelf, so every
 * `?tab=outputs` link (the explorer's Back, Generate's 'Open Deliverables',
 * the backend's own `?tab=outputs&deliverable=<id>`) keeps Play's screen;
 * another tab, a post or a type drill-down still keeps the core's page.
 * P260-FX-23 (M17): the layout hands the switch Play's shelf through its lazy
 * slot, so the shelf arrives a moment after the first render (findBy*).
 * P260-FX-25 (M69): the width comes from useThemeLayout (stood in for here);
 * while it isn't known (the server, hydration) neither the shelf nor the page
 * and its deep link mount.
 * PRD-261 A-04b: on the server the layout adds no Suspense boundary of its own
 * around the page (components/theme-host/theme-slot.tsx: a boundary around it
 * streamed the page twice); the deep link keeps the one it has always had.
 */
import React, { Suspense } from 'react'
import { renderToString } from 'react-dom/server'
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

const state = vi.hoisted(() => ({ play: true, narrow: false, widthUnknown: false, pathname: '/deliverables', search: '' }))

vi.mock('next/navigation', () => ({
  usePathname: () => state.pathname,
  useSearchParams: () => new URLSearchParams(state.search),
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
vi.mock('@/components/play/deliverables/play-shelf', () => ({ PlayShelf: () => <div data-testid="play-shelf" /> }))
vi.mock('@/components/deliverables/deliverable-deep-link', () => ({
  DeliverableDeepLink: () => <div data-testid="deep-link" />,
}))

import DeliverablesLayout from '@/app/deliverables/layout'
import { DeliverableDeepLink } from '@/components/deliverables/deliverable-deep-link'

function CorePage() {
  return <div data-testid="page">The Deliverables page</div>
}

function layoutTree() {
  return (
    <DeliverablesLayout>
      <CorePage />
    </DeliverablesLayout>
  )
}

function renderLayout() {
  return render(layoutTree())
}

async function showsShelf(yes: boolean) {
  if (yes) {
    expect(await screen.findByTestId('play-shelf')).toBeInTheDocument()
    expect(screen.queryByTestId('page')).toBeNull()
    expect(screen.queryByTestId('deep-link')).toBeNull()
  } else {
    expect(screen.getByTestId('page')).toBeInTheDocument()
    expect(screen.queryByTestId('play-shelf')).toBeNull()
    expect(screen.getByTestId('deep-link')).toBeInTheDocument()
  }
}

afterEach(() => {
  cleanup()
  Object.assign(state, { play: true, narrow: false, widthUnknown: false, pathname: '/deliverables', search: '' })
})

describe('app/deliverables/layout.tsx', () => {
  it("renders Play's shelf instead of the page under Play at desktop width, without the deep link's sheet", async () => {
    renderLayout()
    await showsShelf(true)
  })

  it("keeps Play's screen for a link that opens one Deliverable, and the deep link stays closed (K8)", async () => {
    state.search = 'deliverable=d-laptop'
    renderLayout()
    await showsShelf(true)
  })

  it('renders the existing page under Classic or Studio', async () => {
    state.play = false
    renderLayout()
    await showsShelf(false)
  })

  it('renders the existing page below 1024 px (D7)', async () => {
    state.narrow = true
    renderLayout()
    await showsShelf(false)
  })

  it('mounts neither the shelf nor the page and its deep link while the width is not known (M69)', () => {
    state.widthUnknown = true
    state.search = 'deliverable=d-laptop'
    renderLayout()
    // The shelf's chunk is loaded by now (the tests above), so it would render in this pass.
    expect(screen.queryByTestId('play-shelf')).toBeNull()
    expect(screen.queryByTestId('page')).toBeNull()
    expect(screen.queryByTestId('deep-link')).toBeNull()
  })

  it("keeps Studio's explorer inside Play's frame", async () => {
    state.pathname = '/deliverables/explorer'
    renderLayout()
    await showsShelf(false)
  })

  it("keeps Play's shelf for ?tab=outputs, the tab the shelf is (M13)", async () => {
    state.search = 'tab=outputs'
    renderLayout()
    await showsShelf(true)
  })

  it("opens the backend's own Deliverable link (?tab=outputs&deliverable=<id>) as Play's page, the deep link not mounted (K8)", async () => {
    state.search = 'tab=outputs&deliverable=d-laptop'
    renderLayout()
    await showsShelf(true)
  })

  it.each([
    'tab=brand',
    'tab=socials&post=p1',
    'artifact_type=code',
    'tab=outputs&artifact_type=code',
    'tab=outputs&post=p1',
  ])(
    "keeps the core's page for ?%s under Play (D8)",
    async (search) => {
      state.search = search
      renderLayout()
      await showsShelf(false)
    },
  )
})

describe('app/deliverables/layout.tsx on the server', () => {
  it('sends the page and its deep link under Classic or Studio exactly as they render without ThemeSlot', () => {
    state.play = false
    const core = renderToString(
      <>
        <CorePage />
        <Suspense fallback={null}>
          <DeliverableDeepLink />
        </Suspense>
      </>,
    )
    expect(renderToString(layoutTree())).toBe(core)
  })

  it('sends nothing under Play while the width is not known: no page, no deep link, no empty boundary', () => {
    state.widthUnknown = true
    expect(renderToString(layoutTree())).toBe('')
  })
})
