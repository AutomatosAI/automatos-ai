// @ts-check
// PRD-261 A-01: the hello theme's entry. It reaches the browser only through the browser lane's
// route interception (support/hello-theme.ts). Its bare imports resolve through the host's import
// map to the host's own React, ReactDOM and theme SDK, so its state, effects and portal run on
// the instance the App Router renders with.
import { jsx, jsxs } from 'react/jsx-runtime'
import { useEffect, useState } from 'react'
import { createPortal } from 'react-dom'
import { defineTheme } from '@automatos/theme-sdk'

/** @param {{ children?: import('react').ReactNode }} props */
function HelloFrame({ children }) {
  const [count, setCount] = useState(0)
  const [mounted, setMounted] = useState(false)
  useEffect(() => {
    setMounted(true)
  }, [])
  const bar = jsxs('div', {
    'data-hello-bar': '',
    className: 'hello-bar',
    children: [
      jsx('strong', { children: 'Hello theme' }),
      jsx('button', {
        type: 'button',
        'data-testid': 'hello-theme-count',
        onClick: () => setCount((value) => value + 1),
        children: `Clicks: ${count}`,
      }),
    ],
  })
  const badge = jsx('div', { 'data-testid': 'hello-theme-portal', className: 'hello-portal', children: 'Hello theme' })
  return jsxs('div', {
    'data-testid': 'hello-theme-frame',
    className: 'hello-frame',
    children: [
      bar,
      jsx('div', { 'data-testid': 'hello-theme-content', className: 'hello-content', children }),
      mounted ? createPortal(badge, document.body) : null,
    ],
  })
}

export default defineTheme({
  register(host) {
    host.registerSlot('frame@1', HelloFrame)
  },
  unregister() {},
})
