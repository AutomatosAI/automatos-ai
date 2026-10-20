// @ts-check
// PRD-261 A-04b: the theme-route hello theme. Besides its frame, which only wraps the page, it
// registers one route, `hello`, which the host serves at /x/hello (app/x/[...path]/page.tsx,
// through ThemeSlotRoute). It reaches the browser only through the lane's route interception
// (e2e/browser/specs/theme-routes.spec.ts, through support/hello-theme.ts), and its bare imports
// resolve through the host's import map.
import { jsx } from 'react/jsx-runtime'
import { defineTheme } from '@automatos/theme-sdk'

/** @param {{ children?: import('react').ReactNode }} props */
function RouteFrame({ children }) {
  return jsx('div', { 'data-testid': 'hello-a04b-frame', className: 'hello-a04b-frame', children })
}

function HelloRoute() {
  return jsx('section', {
    'data-testid': 'hello-a04b-route',
    className: 'hello-a04b-route',
    'aria-label': 'Hello route',
    children: 'Hello from a theme route',
  })
}

export default defineTheme({
  register(host) {
    host.registerSlot('frame@1', RouteFrame)
    host.registerRoute('hello', HelloRoute)
  },
  unregister() {},
})
