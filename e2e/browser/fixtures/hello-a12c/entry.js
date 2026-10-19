// @ts-check
// PRD-261 A-12c: the hostile theme's entry. Its frame only wraps the page; the attack is in
// theme.css. It reaches the browser only through the browser lane's route interception
// (support/hello-theme.ts serveInstalledTheme), and its bare imports resolve through the host's
// import map to the host's own React and theme SDK.
import { jsx } from 'react/jsx-runtime'
import { defineTheme } from '@automatos/theme-sdk'

/** @param {{ children?: import('react').ReactNode }} props */
function HostileFrame({ children }) {
  return jsx('div', { 'data-testid': 'hello-a12c-frame', className: 'hello-a12c-frame', children })
}

export default defineTheme({
  register(host) {
    host.registerSlot('frame@1', HostileFrame)
  },
  unregister() {},
})
