// PRD-261 A-06a: the hello theme's entry, as an .autotheme package ships it (dist/entry.js).
// Its bare imports resolve through the host's import map to the host's own React and theme
// SDK. It registers the frame, the look editor and the style sheet every theme ships (PA-14),
// and one theme-only page at /x/hello when the host can serve routes (A-04a on).
import { jsx, jsxs } from 'react/jsx-runtime'
import { useState } from 'react'
import { defineTheme } from '@automatos/theme-sdk'

function HelloFrame({ children }) {
  const [count, setCount] = useState(0)
  const bar = jsxs('div', {
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
  return jsxs('div', {
    'data-testid': 'hello-theme-frame',
    className: 'hello-frame',
    children: [bar, jsx('div', { 'data-testid': 'hello-theme-content', className: 'hello-content', children })],
  })
}

function HelloLookEditor() {
  return jsx('section', { 'data-testid': 'hello-look-editor', className: 'hello-content', children: 'Hello look editor' })
}

function HelloStyleSheet() {
  return jsx('section', { 'data-testid': 'hello-style-sheet', className: 'hello-content', children: 'Hello style sheet' })
}

function HelloPage() {
  return jsx('section', { 'data-testid': 'hello-route', className: 'hello-content', children: 'Hello from /x/hello' })
}

export default defineTheme({
  register(host) {
    host.registerSlot('frame@1', HelloFrame)
    host.registerSlot('lookEditor@1', HelloLookEditor)
    host.registerSlot('styleSheet@1', HelloStyleSheet)
    if (typeof host.registerRoute === 'function') host.registerRoute('hello', HelloPage)
  },
  unregister() {},
})
