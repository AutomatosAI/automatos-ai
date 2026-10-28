/**
 * PRD-264 EUAI-11a — the AI label's protected copy keeps itself on screen, whatever a theme's code
 * does to it (protected-caption.tsx): removed or moved, its words or its host's guard changed, hidden
 * with hidePopover(), or covered by a popover shown later. Each is undone at once, with one
 * theme-host warning per kind of tampering, and a theme that undoes every repair gets a few
 * immediate ones a second and the watchdog's after that, so the page never spins. Long agent names
 * are cut so " is an AI" is never lost, and the copy wraps inside the viewport.
 *
 * jsdom has no top layer: ai-label-layout.ts's popover stand-in keeps a state per element and fires
 * `toggle` a microtask later, coalesced, as a browser does. The copy's shadow root is closed; the
 * tests read it off a spy on attachShadow.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  HEAL_WINDOW_MS,
  MAX_HEALS_PER_WINDOW,
  type ProtectedCaption,
  aiLabelCopyTamperedWarning,
  mountProtectedCaption,
} from '@/components/ai-label/protected-caption'
import { RECHECK_INTERVAL_MS } from '@/components/ai-label/use-ai-label-check'
import { ELLIPSIS, IS_AN_AI, MAX_AGENT_NAME_CHARS, aiLabelText } from '@/lib/ai-label/copy'

import {
  type PopoverStandIn,
  captionHost,
  flushMicrotasks,
  installLabelLayout,
  installPopover,
  theCaption,
} from '@/lib/theme-host/__tests__/ai-label-layout'

const WORDS = 'Auto is an AI'
const LOG = '[theme-host]'
const LABEL_BOX = { left: 880, top: 740, width: 110, height: 20 }
const PLACE = () => ({ bottom: 11, right: 23 })

let copy: ProtectedCaption
let popover: PopoverStandIn
let uninstallLayout: () => void

function warnings(): unknown[][] {
  return vi.mocked(console.warn).mock.calls.filter(([prefix]) => prefix === LOG)
}

/** The element showPopover last put in the (pretend) top layer. */
function lastRaised(): Element | undefined {
  return popover.raised[popover.raised.length - 1]
}

function toggle(type: 'toggle' | 'beforetoggle', newState: 'open' | 'closed'): Event {
  return Object.assign(new Event(type), { newState })
}

beforeEach(() => {
  ;({ uninstall: uninstallLayout } = installLabelLayout(LABEL_BOX))
  popover = installPopover()
  vi.spyOn(console, 'warn').mockImplementation(() => undefined)
})

afterEach(() => {
  copy?.remove()
  popover.uninstall()
  uninstallLayout()
  vi.useRealTimers()
  vi.restoreAllMocks()
})

describe("the AI label's protected copy keeps itself on screen (PRD-264 EUAI-11a)", () => {
  it('draws the words in a closed shadow root on a host at the end of <body>, raised into the top layer', () => {
    copy = mountProtectedCaption(document, WORDS, PLACE)
    const host = captionHost() as HTMLElement
    expect(host.parentNode).toBe(document.body)
    expect(host.shadowRoot).toBeNull()
    expect(theCaption()).toHaveTextContent(WORDS)
    expect(popover.raised).toEqual([theCaption()])
    expect(warnings()).toEqual([])
  })

  it('puts the copy back at once when something removes it from <body>, or moves it elsewhere', async () => {
    copy = mountProtectedCaption(document, WORDS, PLACE)
    const first = captionHost() as HTMLElement
    first.remove()
    await flushMicrotasks()
    const second = captionHost() as HTMLElement
    expect(second).not.toBe(first)
    expect(theCaption()).toHaveTextContent(WORDS)
    const elsewhere = document.body.appendChild(document.createElement('div'))
    elsewhere.appendChild(second)
    await flushMicrotasks()
    expect(captionHost()?.parentNode).toBe(document.body)
    expect(elsewhere.childElementCount).toBe(0)
    expect(lastRaised()).toBe(theCaption())
    expect(warnings()).toEqual([[LOG, aiLabelCopyTamperedWarning('removed')]])
    elsewhere.remove()
  })

  it('shows the words again when something hides them with hidePopover()', async () => {
    copy = mountProtectedCaption(document, WORDS, PLACE)
    const hidden = theCaption()
    hidden.hidePopover()
    await flushMicrotasks()
    expect(theCaption()).not.toBe(hidden)
    expect(lastRaised()).toBe(theCaption())
    expect(warnings()).toEqual([[LOG, aiLabelCopyTamperedWarning('hidden')]])
  })

  it('restores the words when something edits them inside the shadow root', async () => {
    copy = mountProtectedCaption(document, WORDS, PLACE)
    theCaption().textContent = 'Auto is a person'
    await flushMicrotasks()
    expect(theCaption()).toHaveTextContent(WORDS)
    expect(theCaption().getAttribute('aria-label')).toBe(WORDS)
    expect(warnings()).toEqual([[LOG, aiLabelCopyTamperedWarning('changed')]])
  })

  it("restores the host's inline guard when something rewrites it", async () => {
    copy = mountProtectedCaption(document, WORDS, PLACE)
    const host = captionHost() as HTMLElement
    host.style.setProperty('display', 'none', 'important')
    await flushMicrotasks()
    const restored = captionHost() as HTMLElement
    expect(restored).not.toBe(host)
    expect(restored.style.getPropertyValue('display')).toBe('block')
    expect(restored.style.getPropertyPriority('display')).toBe('important')
  })

  it('raises the words again after a popover or dialog of the page enters the top layer', async () => {
    copy = mountProtectedCaption(document, WORDS, PLACE)
    const theirs = document.body.appendChild(document.createElement('div'))
    const before = popover.raised.length
    theirs.dispatchEvent(toggle('toggle', 'open'))
    expect(popover.raised).toHaveLength(before + 1)
    theirs.dispatchEvent(toggle('beforetoggle', 'open'))
    await flushMicrotasks()
    expect(popover.raised).toHaveLength(before + 2)
    expect(lastRaised()).toBe(theCaption())
    // one closing, and the copy's own toggles (inside its shadow root), raise nothing more
    theirs.dispatchEvent(toggle('toggle', 'closed'))
    await flushMicrotasks()
    expect(popover.raised).toHaveLength(before + 2)
    expect(warnings()).toEqual([])
    theirs.remove()
  })

  it('raises the words on every watchdog tick, for a cover no event reported', () => {
    vi.useFakeTimers()
    copy = mountProtectedCaption(document, WORDS, PLACE)
    const before = popover.raised.length
    vi.advanceTimersByTime(RECHECK_INTERVAL_MS * 2)
    expect(popover.raised).toHaveLength(before + 2)
  })

  it("can't be made to spin: past MAX_HEALS_PER_WINDOW repairs a second, the watchdog repairs", async () => {
    vi.useFakeTimers()
    copy = mountProtectedCaption(document, WORDS, PLACE)
    for (let repair = 0; repair < MAX_HEALS_PER_WINDOW; repair += 1) {
      captionHost()?.remove()
      await flushMicrotasks()
      expect(captionHost()).not.toBeNull()
    }
    captionHost()?.remove()
    await flushMicrotasks()
    expect(captionHost()).toBeNull()
    vi.advanceTimersByTime(Math.max(RECHECK_INTERVAL_MS, HEAL_WINDOW_MS))
    expect(captionHost()).not.toBeNull()
    expect(theCaption()).toHaveTextContent(WORDS)
  })

  it('leaves nothing behind once removed', async () => {
    copy = mountProtectedCaption(document, WORDS, PLACE)
    copy.remove()
    await flushMicrotasks()
    expect(captionHost()).toBeNull()
  })
})

describe('long agent names (PRD-264 EUAI-11a)', () => {
  it(`cuts a name past ${MAX_AGENT_NAME_CHARS} characters, so "${IS_AN_AI.trim()}" is never lost`, () => {
    const text = aiLabelText('A'.repeat(200))
    expect(text.endsWith(IS_AN_AI)).toBe(true)
    expect(text).toBe(`${'A'.repeat(MAX_AGENT_NAME_CHARS)}${ELLIPSIS}${IS_AN_AI}`)
    expect(aiLabelText('Research Desk')).toBe('Research Desk is an AI')
  })

  it('counts characters, not UTF-16 units, so a cut never splits one', () => {
    const name = '🤖'.repeat(MAX_AGENT_NAME_CHARS + 5)
    expect(aiLabelText(name)).toBe(`${'🤖'.repeat(MAX_AGENT_NAME_CHARS)}${ELLIPSIS}${IS_AN_AI}`)
  })

  it('lets the copy wrap inside the viewport instead of running off it', () => {
    copy = mountProtectedCaption(document, aiLabelText('B'.repeat(200)), PLACE)
    const style = theCaption().style
    expect(style.getPropertyValue('white-space')).toBe('normal')
    expect(style.getPropertyValue('max-width')).toMatch(/^calc\(100vw/)
  })
})
