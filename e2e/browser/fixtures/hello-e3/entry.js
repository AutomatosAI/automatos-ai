// PRD-261 E3c: the storage hello theme's entry, packed as dist/entry.js by ./build.py. Its frame keeps
// one value through the SDK's storage area (useThemeState): the host says whose state it is (this
// theme, the signed-in user) and keeps it in this browser. The theme never touches the browser's
// storage itself. It registers the frame, the look editor and the style sheet every theme ships
// (PA-14), and the theme-only page its theme.json names, as the hello package's entry does.
import { jsx, jsxs } from 'react/jsx-runtime'
import { useState } from 'react'
import { defineTheme, useThemeState } from '@automatos/theme-sdk'

const GREETING_KEY = 'greeting'
const GREETING = 'kept by hello-e3'

/** The greeting this theme keeps, and a button that keeps it; every attribute is for the spec. */
function StoredGreeting() {
  const greeting = useThemeState(GREETING_KEY)
  const [lastWrite, setLastWrite] = useState('none')
  const save = () => {
    const result = greeting.set(GREETING)
    setLastWrite(result.ok ? 'ok' : result.reason)
  }
  return jsxs('div', {
    'data-testid': 'hello-e3-state',
    'data-ready': greeting.ready ? 'true' : 'false',
    'data-value': JSON.stringify(greeting.value),
    'data-persisted': greeting.persisted ? 'true' : 'false',
    'data-last-write': lastWrite,
    children: [
      jsx('span', { children: greeting.ready ? `Kept: ${JSON.stringify(greeting.value)}` : 'Theme state is not ready' }),
      jsx('button', { type: 'button', 'data-testid': 'hello-e3-save', onClick: save, children: 'Keep the greeting' }),
    ],
  })
}

function HelloStorageFrame({ children }) {
  const bar = jsxs('div', {
    className: 'hello-bar',
    children: [jsx('strong', { children: 'Hello storage' }), jsx(StoredGreeting, {})],
  })
  return jsxs('div', {
    'data-testid': 'hello-e3-frame',
    className: 'hello-frame',
    children: [bar, jsx('div', { 'data-testid': 'hello-e3-content', className: 'hello-content', children })],
  })
}

function HelloLookEditor() {
  return jsx('section', { className: 'hello-content', children: 'Hello storage look editor' })
}

function HelloStyleSheet() {
  return jsx('section', { className: 'hello-content', children: 'Hello storage style sheet' })
}

function HelloPage() {
  return jsx('section', { className: 'hello-content', children: 'Hello storage from /x/hello' })
}

export default defineTheme({
  register(host) {
    host.registerSlot('frame@1', HelloStorageFrame)
    host.registerSlot('lookEditor@1', HelloLookEditor)
    host.registerSlot('styleSheet@1', HelloStyleSheet)
    if (typeof host.registerRoute === 'function') host.registerRoute('hello', HelloPage)
  },
  unregister() {},
})
