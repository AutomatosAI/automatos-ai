// @vitest-environment node
import { describe, expect, it } from 'vitest'

// PRD-239 S7 regression: the chat page is server-rendered, and xterm.js reads
// `self` when it is imported. Importing the Canvas module without a browser
// global must not throw — the terminal pane loads client-side only.
describe('Code Canvas server-render safety', () => {
  it('imports without a browser global (xterm is loaded on the client only)', async () => {
    expect(typeof globalThis.window).toBe('undefined')
    await expect(import('./index')).resolves.toHaveProperty('CodingCanvasWidget')
  })
})
