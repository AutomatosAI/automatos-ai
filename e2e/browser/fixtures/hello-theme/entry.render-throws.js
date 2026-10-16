// @ts-check
// PRD-261 A-01: a hello-theme entry whose frame registers cleanly and then throws while drawing.
// The host's boundary must report reason 'render', drop the theme and keep Studio.
import { defineTheme } from '@automatos/theme-sdk'

/** @param {{ children?: import('react').ReactNode }} props */
function BrokenFrame(props) {
  throw new Error(`hello theme: deliberate render failure (${props.children ? 'with' : 'without'} a page)`)
}

export default defineTheme({
  register(host) {
    host.registerSlot('frame@1', BrokenFrame)
  },
})
