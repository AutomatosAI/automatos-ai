// @ts-check
// PRD-261 A-04d: the theme-pages hello theme with no pages: its frame only, which wraps the page.
// Served in place of entry.js (support/hello-theme.ts `overrides`), so every Q3 route renders the
// core page, unchanged, inside the theme's frame.
import { jsx } from 'react/jsx-runtime'
import { defineTheme } from '@automatos/theme-sdk'

/** @param {{ children?: import('react').ReactNode }} props */
function PagesFrame({ children }) {
  return jsx('div', { 'data-testid': 'hello-a04d-frame', className: 'hello-a04d-frame', children })
}

export default defineTheme({
  register(host) {
    host.registerSlot('frame@1', PagesFrame)
  },
  unregister() {},
})
