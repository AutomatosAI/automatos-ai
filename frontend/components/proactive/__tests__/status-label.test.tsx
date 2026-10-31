/**
 * PRD-268 §6.11.1, §6.12.2 (US-C21c) — StatusLabel draws the glyph and the word for every state,
 * with ▸ next_step and Try again when the row can be retried: a studio-pill painted by the role's
 * token per ground tone in Studio, Classic's status badge in Classic, never colour alone, and
 * nothing for an unknown state.
 */
import { fireEvent, render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const look = vi.hoisted(() => ({ studio: true }))
vi.mock('@/hooks/use-studio-theme', () => ({ useIsStudio: () => look.studio }))

import { CLASSIC_ROLE_VARIANT, STUDIO_ROLE_CLASS, StatusLabel } from '@/components/proactive/status-label'
import { DISPLAY_STATE_LABELS, STATUS_ROLE_TOKENS } from '@/lib/proactive/display-state'
import { DISPLAY_STATES } from '@/lib/proactive/types'
import { HEX_COLOUR, MAX_COMPONENT_LINES, ORANGE_CLASS, ROUNDED_3XL, componentSpans, proactiveSource } from './source-rules'

/** The pill: the element that holds the state's word. */
function pillOf(word: string): HTMLElement {
  const pill = screen.getByText(word).parentElement
  if (pill === null) throw new Error(`no pill for ${word}`)
  return pill
}

describe('StatusLabel in Studio', () => {
  beforeEach(() => {
    look.studio = true
  })

  it.each(DISPLAY_STATES)('draws %s as a studio-pill with its glyph and its word', (state) => {
    const label = DISPLAY_STATE_LABELS[state]
    render(<StatusLabel state={state} />)
    const pill = pillOf(label.word)
    expect(pill).toHaveClass('studio-pill')
    expect(pill).toHaveTextContent(`${label.glyph}${label.word}`)
    // The glyph is decoration for a screen reader: the word says it.
    expect(screen.getByText(label.glyph)).toHaveAttribute('aria-hidden', 'true')
  })

  it.each(DISPLAY_STATES)("paints %s with its role's token on a light ground and on a dark one", (state) => {
    const { role } = DISPLAY_STATE_LABELS[state]
    const token = STATUS_ROLE_TOKENS[role]
    render(<StatusLabel state={state} />)
    const classes = pillOf(DISPLAY_STATE_LABELS[state].word).className
    expect(classes).toContain(STUDIO_ROLE_CLASS[role])
    expect(STUDIO_ROLE_CLASS[role]).toMatch(new RegExp(`(^|\\s)text-\\[hsl\\(var\\(${token.light}\\)\\)\\]`))
    expect(STUDIO_ROLE_CLASS[role]).toContain(`dark:text-[hsl(var(${token.dark}))]`)
  })

  it('draws needs-you in warn-ink on a light ground and warning on a dark one', () => {
    expect(STUDIO_ROLE_CLASS.needs_you).toBe('text-[hsl(var(--warn-ink))] dark:text-[hsl(var(--warning))]')
  })

  it('never paints a status through --accent, and never on a tinted fill', () => {
    Object.values(STUDIO_ROLE_CLASS).forEach((classes) => {
      expect(classes).not.toContain('--accent')
      expect(classes).not.toMatch(/\bbg-/)
    })
  })
})

describe('StatusLabel in Classic', () => {
  beforeEach(() => {
    look.studio = false
  })

  it.each(DISPLAY_STATES)("draws %s as Classic's status badge with its glyph and its word", (state) => {
    const label = DISPLAY_STATE_LABELS[state]
    render(<StatusLabel state={state} />)
    const badge = pillOf(label.word)
    expect(badge).not.toHaveClass('studio-pill')
    expect(badge).toHaveClass('rounded-full')
    expect(badge).toHaveTextContent(`${label.glyph}${label.word}`)
  })

  it("uses Classic's badge tokens per role: warning, info for working, success, destructive", () => {
    expect(CLASSIC_ROLE_VARIANT).toMatchObject({ needs_you: 'warning', working: 'active', ready: 'success', done: 'success', failed: 'error' })
    render(<StatusLabel state="failed" />)
    expect(pillOf("Couldn't finish").className).toContain('--destructive')
  })
})

describe('StatusLabel parts', () => {
  beforeEach(() => {
    look.studio = true
  })

  it.each([undefined, null, '', 'unknown', 'WORKING', 'stopped', 3, {}])('draws nothing for %p', (state) => {
    const { container } = render(<StatusLabel state={state} nextStep="Looking at it" retry="retry_ticket" onRetry={vi.fn()} />)
    expect(container).toBeEmptyDOMElement()
  })

  it('shows the running step after ▸', () => {
    render(<StatusLabel state="working" nextStep="42 of 60 made" />)
    expect(screen.getByText('42 of 60 made')).toHaveTextContent('▸ 42 of 60 made')
  })

  it('shows Try again when the row says what retry means, named by the row, and runs it', () => {
    const onRetry = vi.fn()
    render(<StatusLabel state="failed" retry="rerun_mission" onRetry={onRetry} title="Lisbon flights" />)
    fireEvent.click(screen.getByRole('button', { name: 'Try again: Lisbon flights' }))
    expect(onRetry).toHaveBeenCalledWith('rerun_mission')
  })

  it('draws no Try again without a retry, with an unknown retry, or with nothing to run it', () => {
    const { rerender } = render(<StatusLabel state="failed" onRetry={vi.fn()} />)
    expect(screen.queryByRole('button')).toBeNull()
    rerender(<StatusLabel state="failed" retry={'reboot' as never} onRetry={vi.fn()} />)
    expect(screen.queryByRole('button')).toBeNull()
    rerender(<StatusLabel state="failed" retry="retry_ticket" />)
    expect(screen.queryByRole('button')).toBeNull()
  })

  it('never shows a status by colour alone: every state carries a word', () => {
    DISPLAY_STATES.forEach((state) => {
      const { container, unmount } = render(<StatusLabel state={state} />)
      expect(container.textContent).toContain(DISPLAY_STATE_LABELS[state].word)
      unmount()
    })
  })
})

describe('status-label.tsx source', () => {
  const source = proactiveSource('status-label.tsx')

  it('uses tokens only: no hex, no orange, no rounded-3xl', () => {
    expect(source).not.toMatch(HEX_COLOUR)
    expect(source).not.toMatch(ORANGE_CLASS)
    expect(source).not.toMatch(ROUNDED_3XL)
  })

  it(`keeps each component at most ${MAX_COMPONENT_LINES} lines`, () => {
    const spans = componentSpans(source)
    expect(spans.map((span) => span.name)).toContain('StatusLabel')
    spans.forEach((span) => expect(span.lines, span.name).toBeLessThanOrEqual(MAX_COMPONENT_LINES))
  })
})
