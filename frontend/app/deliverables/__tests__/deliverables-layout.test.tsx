/**
 * PRD-260 T-007a, K7, D7, D8 — app/deliverables/layout.tsx mounts Play's
 * shelf through PlayRouteSwitch: under Play at >= 1024 px on /deliverables
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
 */
import React from 'react'
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

const state = vi.hoisted(() => ({ play: true, narrow: false, pathname: '/deliverables', search: '' }))

vi.mock('next/navigation', () => ({
  usePathname: () => state.pathname,
  useSearchParams: () => new URLSearchParams(state.search),
}))
vi.mock('@/hooks/use-mobile', () => ({ useIsTabletOrBelow: () => state.narrow, useIsMobile: () => state.narrow }))
vi.mock('@/hooks/use-studio-theme', () => ({ useIsStudio: () => true, useIsPlay: () => state.play }))
vi.mock('@/components/play/deliverables/play-shelf', () => ({ PlayShelf: () => <div data-testid="play-shelf" /> }))
vi.mock('@/components/deliverables/deliverable-deep-link', () => ({
  DeliverableDeepLink: () => <div data-testid="deep-link" />,
}))

import DeliverablesLayout from '@/app/deliverables/layout'

function renderLayout() {
  return render(
    <DeliverablesLayout>
      <div data-testid="page">The Deliverables page</div>
    </DeliverablesLayout>,
  )
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
  Object.assign(state, { play: true, narrow: false, pathname: '/deliverables', search: '' })
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
