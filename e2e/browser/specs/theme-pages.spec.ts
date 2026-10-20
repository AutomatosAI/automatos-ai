import { expect, test, type Locator, type Page } from '@playwright/test'
import { chooseStyle, serveInstalledTheme, type InstalledTheme } from '../support/hello-theme'
import { watchPage } from '../support/watch'

// PRD-261 A-04d (@host): a theme page on every page of Gerard's Q3 list, through ThemeSlot.
//
// The hello theme (fixtures/hello-a04d) registers its own page on every route of the host's page
// list: chat, deliverables, missions (/assignments and /missions/[id]), agents, knowledge
// (/documents), playbooks, the Command Center and home (/).
//
// 1. With entry.js, each route renders the theme's page (a section naming its route) inside the
//    theme's frame, and never the core page; home stays on / instead of forwarding to /chat.
// 2. With entry.frame-only.js (the frame, no pages), each route renders the core page, unchanged,
//    inside the same frame; home forwards to /chat, and /playbooks to the Assignments hub as it
//    does under Studio.

const APP_URL = process.env.E2E_APP_URL ?? 'http://localhost:3000'
const STATUS_ATTR = 'data-theme-host'
const FRAME_TEST_ID = 'hello-a04d-frame'
const PAGE_TEST_ID = 'hello-a04d-page'
const ROUTE_ATTR = 'data-route'

const WITH_PAGES: InstalledTheme = { installId: '00000000-0000-4000-8000-0000000a04d0', fixture: 'hello-a04d' }
const FRAME_ONLY: InstalledTheme = {
  installId: '00000000-0000-4000-8000-0000000a04d1',
  fixture: 'hello-a04d',
  overrides: { 'entry.js': 'entry.frame-only.js' },
}

/** A mission no workspace has: the core page stays on its loading skeleton. */
const UNKNOWN_MISSION = '00000000-0000-4000-8000-00000000a04d'
/** The core chat's composer (components/chatbot/multimodal-input.tsx). */
const CORE_COMPOSER = 'textarea[placeholder="Send a message..."]'
/** The core's loading skeleton (components/ui/skeleton.tsx). */
const CORE_SKELETON = '.animate-pulse'

interface QPage {
  /** Gerard's Q3 name. */
  readonly name: string
  /** The address opened. */
  readonly address: string
  /** The route the theme registers (THEME_PAGE_ROUTES). */
  readonly route: string
  /** Where the core page lands: the address itself, or where it forwards. */
  readonly corePath: RegExp
  /** Something only the core page draws there. */
  readonly core: (frame: Locator) => Locator
}

function heading(name: RegExp) {
  return (frame: Locator) => frame.getByRole('heading', { level: 1, name })
}

function coreComposer(frame: Locator): Locator {
  return frame.locator(CORE_COMPOSER).filter({ visible: true }).first()
}

const Q3_PAGES: readonly QPage[] = [
  { name: 'chat', address: '/chat', route: '/chat', corePath: /\/chat$/, core: coreComposer },
  {
    name: 'deliverables',
    address: '/deliverables',
    route: '/deliverables',
    corePath: /\/deliverables$/,
    core: heading(/Deliverables/),
  },
  {
    name: 'missions (Assignments)',
    address: '/assignments?tab=missions',
    route: '/assignments',
    corePath: /\/assignments\?tab=missions$/,
    core: heading(/^Assignments$/),
  },
  {
    name: 'missions (one Mission)',
    address: `/missions/${UNKNOWN_MISSION}`,
    route: '/missions/[id]',
    corePath: new RegExp(`/missions/${UNKNOWN_MISSION}$`),
    core: (frame) => frame.locator(CORE_SKELETON).first(),
  },
  { name: 'agents', address: '/agents', route: '/agents', corePath: /\/agents$/, core: heading(/^Agent Management$/) },
  { name: 'knowledge', address: '/documents', route: '/documents', corePath: /\/documents$/, core: heading(/Knowledge/) },
  // Under Studio the core /playbooks forwards to the Assignments hub's Playbooks tab (app/playbooks/page.tsx).
  {
    name: 'playbooks',
    address: '/playbooks',
    route: '/playbooks',
    corePath: /\/assignments\?tab=playbooks$/,
    core: heading(/^Assignments$/),
  },
  {
    name: 'command-center',
    address: '/command-center',
    route: '/command-center',
    corePath: /\/command-center$/,
    core: heading(/^Command Center$/),
  },
  // Home has no core page: it forwards to Chat (app/page.tsx).
  { name: 'home', address: '/', route: '/', corePath: /\/chat$/, core: coreComposer },
]

async function openUnder(page: Page, theme: InstalledTheme, address: string): Promise<Locator> {
  await chooseStyle(page, `theme:install:${theme.installId}`, APP_URL)
  await serveInstalledTheme(page, theme)
  await page.goto(address)
  await expect(page.locator('html')).toHaveAttribute(STATUS_ATTR, 'active')
  const frame = page.getByTestId(FRAME_TEST_ID).first()
  await expect(frame).toBeVisible()
  return frame
}

function pathOf(page: Page): string {
  const url = new URL(page.url())
  return `${url.pathname}${url.search}`
}

test.describe('theme pages on every Q3 page through ThemeSlot', () => {
  for (const q3 of Q3_PAGES) {
    test(`${q3.name}: the theme's registerPage('${q3.route}') renders inside its frame`, { tag: '@host' }, async ({ page }) => {
      const watch = watchPage(page)
      const frame = await openUnder(page, WITH_PAGES, q3.address)

      const themePage = frame.getByTestId(PAGE_TEST_ID)
      await expect(themePage).toHaveCount(1)
      await expect(themePage).toHaveAttribute(ROUTE_ATTR, q3.route)
      await expect(q3.core(frame)).toHaveCount(0)
      expect(pathOf(page)).toBe(q3.address)
      expect(watch.pageErrors).toEqual([])
    })

    test(`${q3.name}: the core page renders unchanged when the theme registers none`, { tag: '@host' }, async ({ page }) => {
      const watch = watchPage(page)
      const frame = await openUnder(page, FRAME_ONLY, q3.address)

      await expect(page).toHaveURL(q3.corePath)
      await expect(q3.core(frame)).toBeVisible()
      await expect(frame.getByTestId(PAGE_TEST_ID)).toHaveCount(0)
      expect(watch.pageErrors).toEqual([])
    })
  }
})
