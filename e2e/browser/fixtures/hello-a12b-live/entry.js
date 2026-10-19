// @ts-check
// PRD-261 A-12b: the placement-slot hello theme. Besides its frame, which only wraps the page, it
// registers one placement slot, page-header.mascot@1, which draws in the shared PageHeader of every
// page that renders one (docs/theme-host/placement-reach.md). It reaches the browser only through
// the lane's route interception (e2e/browser/specs/placement-slots.spec.ts, through
// support/hello-theme.ts), and its bare imports resolve through the host's import map.
import { jsx } from 'react/jsx-runtime'
import { defineTheme } from '@automatos/theme-sdk'

/** @param {{ children?: import('react').ReactNode }} props */
function LiveFrame({ children }) {
  return jsx('div', { 'data-testid': 'hello-a12b-live-frame', className: 'hello-a12b-live-frame', children })
}

/** @param {{ base: string, title: string, titleAccent: string, eyebrow: string | null }} props */
function HeaderMascot({ base, title, titleAccent }) {
  return jsx('span', {
    'data-testid': 'hello-a12b-live-mascot',
    'data-base': base,
    className: 'hello-a12b-live-mascot',
    children: `${title} ${titleAccent}`,
  })
}

export default defineTheme({
  register(host) {
    host.registerSlot('frame@1', LiveFrame)
    host.registerSlot('page-header.mascot@1', HeaderMascot)
  },
  unregister() {},
})
