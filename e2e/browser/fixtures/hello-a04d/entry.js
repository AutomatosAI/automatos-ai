// @ts-check
// PRD-261 A-04d: the theme-pages hello theme. Besides its frame, which only wraps the page, it
// registers its own page on every route of the host's page list (THEME_PAGE_ROUTES, Gerard's Q3
// list: chat, deliverables, missions, agents, knowledge, playbooks, command-center and home), each
// a section naming its route. It reaches the browser only through the lane's route interception
// (e2e/browser/specs/theme-pages.spec.ts, through support/hello-theme.ts), and its bare imports
// resolve through the host's import map. entry.frame-only.js is the same theme with no pages.
import { jsx } from 'react/jsx-runtime'
import { defineTheme } from '@automatos/theme-sdk'

/** Every route of lib/theme-host/constants.ts THEME_PAGE_ROUTES. */
const PAGE_ROUTES = [
  '/chat',
  '/deliverables',
  '/assignments',
  '/missions/[id]',
  '/agents',
  '/documents',
  '/playbooks',
  '/command-center',
  '/',
]

/** @param {{ children?: import('react').ReactNode }} props */
function PagesFrame({ children }) {
  return jsx('div', { 'data-testid': 'hello-a04d-frame', className: 'hello-a04d-frame', children })
}

/** @param {string} route */
function pageFor(route) {
  return function HelloPage() {
    return jsx('section', {
      'data-testid': 'hello-a04d-page',
      'data-route': route,
      className: 'hello-a04d-page',
      'aria-label': 'Hello page',
      children: `Hello from a theme page on ${route}`,
    })
  }
}

export default defineTheme({
  register(host) {
    host.registerSlot('frame@1', PagesFrame)
    for (const route of PAGE_ROUTES) host.registerPage(route, pageFor(route))
  },
  unregister() {},
})
