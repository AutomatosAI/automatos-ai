/**
 * PRD-261 A-19a (P260-FX-22, M20, M21) — the host's lib/theme-host/focus.ts: focus is put back
 * only when it was really dropped to <body>, never taken from where the user or the page put it;
 * refocusWhenShown focuses an element at once, or waits for one rendered late (a theme's screen
 * behind a dynamic import), and stops waiting once its effect is cleaned up. Elements are built
 * with createElement (no innerHTML).
 */
import { waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { focusWasDropped, refocus, refocusWhenShown } from '../focus'

const SELECTOR = '[data-focus-target]'

/** A heading that takes focus from script only, as a theme's screen heading does. */
function target(): HTMLHeadingElement {
  const heading = document.createElement('h1')
  heading.tabIndex = -1
  heading.dataset.focusTarget = ''
  return heading
}

/** Let the MutationObserver's pending records, if any, be delivered. */
const nextTask = () => new Promise((resolve) => setTimeout(resolve, 0))

afterEach(() => {
  document.body.replaceChildren()
})

describe('the host refocus', () => {
  it('focuses the target when focus was dropped to <body>', () => {
    const heading = target()
    document.body.append(heading)
    expect(focusWasDropped()).toBe(true)
    refocus(heading)
    expect(heading).toHaveFocus()
  })

  it('never takes focus from where the user or the page put it', () => {
    const field = document.createElement('input')
    const heading = target()
    document.body.append(field, heading)
    field.focus()
    expect(focusWasDropped()).toBe(false)
    refocus(heading)
    expect(field).toHaveFocus()
  })

  it('counts focus as dropped once the focused element leaves the page', () => {
    const button = document.createElement('button')
    document.body.append(button)
    button.focus()
    expect(focusWasDropped()).toBe(false)
    button.remove()
    expect(focusWasDropped()).toBe(true)
  })

  it('does nothing without a target', () => {
    expect(() => refocus(null)).not.toThrow()
    expect(() => refocus(undefined)).not.toThrow()
    expect(document.body).toHaveFocus()
  })
})

describe('the host refocusWhenShown', () => {
  it('focuses an element already on the page at once', () => {
    const heading = target()
    document.body.append(heading)
    refocusWhenShown(SELECTOR)
    expect(heading).toHaveFocus()
  })

  it('waits for an element rendered later, then focuses it', async () => {
    refocusWhenShown(SELECTOR)
    await nextTask()
    const heading = target()
    document.body.append(heading)
    await waitFor(() => expect(heading).toHaveFocus())
  })

  it('stops waiting once cleaned up: a page left before then never takes focus', async () => {
    const stop = refocusWhenShown(SELECTOR)
    stop()
    const heading = target()
    document.body.append(heading)
    await nextTask()
    expect(heading).not.toHaveFocus()
  })
})
