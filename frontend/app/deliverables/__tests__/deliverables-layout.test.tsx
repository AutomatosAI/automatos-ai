/**
 * PRD-260 T-007a, K7, D7, D8 — app/deliverables/layout.tsx mounts a theme's
 * Deliverables page through ThemeSlot (PRD-261 A-04b: the theme registers its
 * /deliverables page; here the fixture theme's, PRD-261 A-19b, handed over by
 * the theme host): under the theme at >= 1024 px on /deliverables
 * only, the existing page under Classic and Studio, below 1024 px, on the
 * explorer, and for a link into one of the page's other sections (a tab, a
 * post, a type drill-down). DeliverableDeepLink is mounted beside the page
 * only: under the theme its page opens a Deliverable as its own page
 * (T-007b, K8), so the sheet never opens over it.
 *
 * P260-FX-11 (M13): the Outputs tab IS the theme's page, so every
 * `?tab=outputs` link (the explorer's Back, Generate's 'Open Deliverables',
 * the backend's own `?tab=outputs&deliverable=<id>`) keeps the theme's screen;
 * another tab, a post or a type drill-down still keeps the core's page.
 * P260-FX-25 (M69): the width comes from useThemeLayout (stood in for here);
 * while it isn't known (the server, hydration) neither the theme's page nor
 * the core page and its deep link mount.
 * PRD-261 A-04b: on the server the layout adds no Suspense boundary of its own
 * around the page (components/theme-host/theme-slot.tsx: a boundary around it
 * streamed the page twice); the deep link keeps the one it has always had.
 */
import React, { Suspense } from 'react'
import { renderToString } from 'react-dom/server'
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

const state = vi.hoisted(() => ({ themed: true, narrow: false, widthUnknown: false, pathname: '/deliverables', search: '' }))

vi.mock('next/navigation', () => ({
  usePathname: () => state.pathname,
  useSearchParams: () => new URLSearchParams(state.search),
}))
vi.mock('@/hooks/use-studio-theme', async () => {
  const { FIXTURE_THEME_ENTRY } = await import('@/lib/theme-host/__tests__/fixture-theme-entry')
  return {
    useIsStudio: () => true,
    useActiveTheme: () => (state.themed ? FIXTURE_THEME_ENTRY : null),
    useThemeLayout: () => {
      if (!state.themed || state.narrow) return 'fallback'
      return state.widthUnknown ? 'pending' : 'theme'
    },
  }
})
// PRD-261 A-04d: ThemeSlot draws an installed theme's page full bleed inside MainLayout (its frame); it stands in here.
vi.mock('@/components/layout/main-layout', () => ({
  MainLayout: ({ children }: { children?: React.ReactNode }) => <>{children}</>,
}))
vi.mock('@/components/deliverables/deliverable-deep-link', () => ({
  DeliverableDeepLink: () => <div data-testid="deep-link" />,
}))

import DeliverablesLayout from '@/app/deliverables/layout'
import { DeliverableDeepLink } from '@/components/deliverables/deliverable-deep-link'
import { ThemeHostContext } from '@/components/theme-host/theme-host-context'
import { FIXTURE_THEME_TEST_IDS, fixtureThemeHost } from '@/lib/theme-host/__tests__/fixture-theme-entry'

/** The theme's Deliverables page, as the fixture theme registers it. */
const THEME_PAGE = FIXTURE_THEME_TEST_IDS.deliverables

function CorePage() {
  return <div data-testid="page">The Deliverables page</div>
}

/** The layout around the core page; under the theme, inside the theme host that loaded it. */
function layoutTree() {
  const tree = (
    <DeliverablesLayout>
      <CorePage />
    </DeliverablesLayout>
  )
  return state.themed ? <ThemeHostContext.Provider value={fixtureThemeHost()}>{tree}</ThemeHostContext.Provider> : tree
}

function renderLayout() {
  return render(layoutTree())
}

async function showsShelf(yes: boolean) {
  if (yes) {
    expect(await screen.findByTestId(THEME_PAGE)).toBeInTheDocument()
    expect(screen.queryByTestId('page')).toBeNull()
    expect(screen.queryByTestId('deep-link')).toBeNull()
  } else {
    expect(screen.getByTestId('page')).toBeInTheDocument()
    expect(screen.queryByTestId(THEME_PAGE)).toBeNull()
    expect(screen.getByTestId('deep-link')).toBeInTheDocument()
  }
}

afterEach(() => {
  cleanup()
  Object.assign(state, { themed: true, narrow: false, widthUnknown: false, pathname: '/deliverables', search: '' })
})

describe('app/deliverables/layout.tsx', () => {
  it("renders the theme's page instead of the core page under the theme at desktop width, without the deep link's sheet", async () => {
    renderLayout()
    await showsShelf(true)
  })

  it("keeps the theme's screen for a link that opens one Deliverable, and the deep link stays closed (K8)", async () => {
    state.search = 'deliverable=d-laptop'
    renderLayout()
    await showsShelf(true)
  })

  it('renders the existing page under Classic or Studio', async () => {
    state.themed = false
    renderLayout()
    await showsShelf(false)
  })

  it('renders the existing page below 1024 px (D7)', async () => {
    state.narrow = true
    renderLayout()
    await showsShelf(false)
  })

  it("mounts neither the theme's page nor the core page and its deep link while the width is not known (M69)", () => {
    state.widthUnknown = true
    state.search = 'deliverable=d-laptop'
    renderLayout()
    // The theme's page is registered and needs no chunk, so it would render in this pass.
    expect(screen.queryByTestId(THEME_PAGE)).toBeNull()
    expect(screen.queryByTestId('page')).toBeNull()
    expect(screen.queryByTestId('deep-link')).toBeNull()
  })

  it("keeps Studio's explorer inside the theme's frame", async () => {
    state.pathname = '/deliverables/explorer'
    renderLayout()
    await showsShelf(false)
  })

  it("keeps the theme's page for ?tab=outputs, the tab its page is (M13)", async () => {
    state.search = 'tab=outputs'
    renderLayout()
    await showsShelf(true)
  })

  it("opens the backend's own Deliverable link (?tab=outputs&deliverable=<id>) as the theme's page, the deep link not mounted (K8)", async () => {
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
    "keeps the core's page for ?%s under the theme (D8)",
    async (search) => {
      state.search = search
      renderLayout()
      await showsShelf(false)
    },
  )
})

describe('app/deliverables/layout.tsx on the server', () => {
  it('sends the page and its deep link under Classic or Studio exactly as they render without ThemeSlot', () => {
    state.themed = false
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

  it('sends nothing under the theme while the width is not known: no page, no deep link, no empty boundary', () => {
    state.widthUnknown = true
    expect(renderToString(layoutTree())).toBe('')
  })
})
