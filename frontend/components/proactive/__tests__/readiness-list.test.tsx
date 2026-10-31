/**
 * PRD-268 §6.11.3, §6.12.3, K17 (US-C21d) — ReadinessList: each item's label, detail, glyph and
 * word (✓ Ready, ○ Not checked, ◆ Needs setup), its fix as an in-app link only, the client's rows
 * after the server's, each row read as one sentence ("Calendar: needs setup. Connect calendar."),
 * checking while nothing is read or Check again runs, Check again through useCheckReadiness with
 * its answer shown when newer, "Checked a moment ago" on a rate-limited answer, one line for any
 * other refusal, and the same parts in Studio light, Studio dark and Classic.
 */
import { fireEvent, render, screen, within } from '@testing-library/react'
import type { ReactNode } from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/lib/api-client', () => ({ apiClient: {} }))
vi.mock('next/link', () => ({
  default: ({ href, children, ...rest }: { href: string; children: ReactNode; className?: string }) => (
    <a href={href} {...rest}>
      {children}
    </a>
  ),
}))

type Check = { mutate: ReturnType<typeof vi.fn>; isLoading: boolean; isError: boolean; error: unknown; data: unknown }
const hooks = vi.hoisted(() => ({ check: null as unknown as Check, studio: true }))
vi.mock('@/hooks/use-readiness', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@/hooks/use-readiness')>()),
  useCheckReadiness: () => hooks.check,
}))
vi.mock('@/hooks/use-studio-theme', () => ({ useIsStudio: () => hooks.studio }))

import {
  CHECKED_A_MOMENT_AGO,
  CHECKING,
  CHECK_AGAIN,
  READINESS_TITLE,
  ReadinessList,
  newestReadiness,
  readinessSentence,
} from '@/components/proactive/readiness-list'
import { REFUSED_FALLBACK } from '@/components/proactive/use-suggestion-card'
import type { Readiness, ReadinessItem } from '@/lib/proactive/types'
import { HEX_COLOUR, MAX_COMPONENT_LINES, ORANGE_CLASS, ROUNDED_3XL, componentSpans, proactiveSource } from './source-rules'

const CORE: ReadinessItem = { id: 'core', label: 'Automatos', detail: 'localhost:8000 · 12 ms', state: 'ready', needed_by: ['brief'] }
const SEARCH: ReadinessItem = { id: 'web_search', label: 'Web search', detail: 'No answer in time.', state: 'not_checked', needed_by: ['check_in'] }
const CALENDAR: ReadinessItem = {
  id: 'calendar',
  label: 'Calendar',
  detail: 'No calendar connected.',
  state: 'needs_setup',
  fix: { label: 'Connect calendar', href: '/settings?tab=integrations' },
  needed_by: ['brief'],
}
const MICROPHONE: ReadinessItem = { id: 'microphone', label: 'Microphone', detail: 'This browser', state: 'ready', needed_by: ['voice'] }

function readiness(items: ReadinessItem[] = [CORE, SEARCH, CALENDAR], checkedAt = '2026-10-30T08:00:00Z'): Readiness {
  return { v: 1, checked_at: checkedAt, items }
}

function refusal(status: number, body: unknown): Error {
  return Object.assign(new Error(`HTTP ${status}`), { status, body })
}

const rowOf = (container: HTMLElement, id: string) => container.querySelector(`li[data-readiness-id="${id}"]`) as HTMLElement

beforeEach(() => {
  hooks.check = { mutate: vi.fn(), isLoading: false, isError: false, error: null, data: undefined }
  hooks.studio = true
})

describe('ReadinessList rows', () => {
  it("shows each item's label, detail, and its glyph with the word", () => {
    const { container } = render(<ReadinessList readiness={readiness()} />)
    expect(screen.getByRole('region', { name: READINESS_TITLE })).toBeInTheDocument()
    const cases: [string, string, string, string][] = [
      ['core', 'Automatos', '✓', 'Ready'],
      ['web_search', 'Web search', '○', 'Not checked'],
      ['calendar', 'Calendar', '◆', 'Needs setup'],
    ]
    cases.forEach(([id, label, glyph, word]) => {
      const row = within(rowOf(container, id))
      expect(row.getByText(label)).toBeInTheDocument()
      expect(row.getByText(glyph)).toHaveAttribute('aria-hidden', 'true')
      expect(row.getByText(word)).toBeInTheDocument()
    })
    expect(within(rowOf(container, 'web_search')).getByText('No answer in time.')).toBeInTheDocument()
  })

  it('reads each row as one sentence', () => {
    render(<ReadinessList readiness={readiness()} />)
    expect(screen.getByRole('listitem', { name: 'Calendar: needs setup. Connect calendar.' })).toBeInTheDocument()
    expect(screen.getByRole('listitem', { name: 'Web search: not checked.' })).toBeInTheDocument()
    expect(screen.getByRole('listitem', { name: 'Automatos: ready.' })).toBeInTheDocument()
    expect(screen.getByRole('listitem', { name: 'Calendar: needs setup. Connect calendar.' })).toHaveAccessibleDescription('No calendar connected.')
    expect(readinessSentence(CALENDAR)).toBe('Calendar: needs setup. Connect calendar.')
  })

  it('draws the fix as an in-app link', () => {
    render(<ReadinessList readiness={readiness()} />)
    expect(screen.getByRole('link', { name: 'Connect calendar' })).toHaveAttribute('href', '/settings?tab=integrations')
  })

  it.each(['https://evil.example/setup', '//evil.example', 'javascript:alert(1)', '/\\evil.example'])('never links a fix at %s', (href) => {
    const outside = { ...CALENDAR, fix: { label: 'Connect calendar', href } }
    render(<ReadinessList readiness={readiness([outside])} />)
    expect(screen.queryByRole('link')).toBeNull()
    expect(screen.getByRole('listitem', { name: 'Calendar: needs setup.' })).toBeInTheDocument()
  })

  it("draws the client's rows after the server's", () => {
    const { container } = render(<ReadinessList readiness={readiness([CORE])} clientItems={[MICROPHONE]} />)
    const ids = [...container.querySelectorAll('li[data-readiness-id]')].map((row) => row.getAttribute('data-readiness-id'))
    expect(ids).toEqual(['core', 'microphone'])
  })

  it('draws no row for a state it does not know', () => {
    const odd = { ...CORE, id: 'odd', state: 'broken' } as unknown as ReadinessItem
    const { container } = render(<ReadinessList readiness={readiness([odd, CORE])} />)
    expect(rowOf(container, 'odd')).toBeNull()
    expect(rowOf(container, 'core')).not.toBeNull()
  })
})

describe('ReadinessList, Check again', () => {
  it('says it is checking while nothing has been read yet', () => {
    render(<ReadinessList readiness={undefined} />)
    expect(screen.getByRole('status')).toHaveTextContent(CHECKING)
    expect(screen.getByRole('region', { name: READINESS_TITLE })).toHaveAttribute('aria-busy', 'true')
  })

  it('gives a read that failed one line in place of checking, and still offers Check again', () => {
    render(<ReadinessList readiness={undefined} error={refusal(502, { detail: 'pool exhausted on db-1' })} />)
    expect(screen.getByRole('alert')).toHaveTextContent(REFUSED_FALLBACK)
    expect(screen.queryByText(CHECKING)).toBeNull()
    expect(screen.queryByText(/db-1/)).toBeNull()
    expect(screen.getByRole('region', { name: READINESS_TITLE })).toHaveAttribute('aria-busy', 'false')
    expect(screen.getByRole('button', { name: CHECK_AGAIN })).toBeEnabled()
  })

  it('keeps every row when a client row shares an id with a server row', () => {
    const { container } = render(<ReadinessList readiness={readiness([CORE])} clientItems={[{ ...MICROPHONE, id: 'core' }]} />)
    expect(container.querySelectorAll('li[data-readiness-id="core"]')).toHaveLength(2)
  })

  it('posts the check through useCheckReadiness', () => {
    render(<ReadinessList readiness={readiness()} />)
    fireEvent.click(screen.getByRole('button', { name: CHECK_AGAIN }))
    expect(hooks.check.mutate).toHaveBeenCalledTimes(1)
  })

  it('holds Check again and says it is checking while it runs', () => {
    hooks.check = { ...hooks.check, isLoading: true }
    render(<ReadinessList readiness={readiness()} />)
    const button = screen.getByRole('button', { name: CHECKING })
    expect(button).toBeDisabled()
    expect(button).toHaveAttribute('aria-busy', 'true')
  })

  it("shows Check again's answer when it is newer than the one passed", () => {
    hooks.check = { ...hooks.check, data: readiness([{ ...CALENDAR, state: 'ready', fix: null }], '2026-10-30T09:00:00Z') }
    render(<ReadinessList readiness={readiness()} />)
    expect(screen.getByRole('listitem', { name: 'Calendar: ready.' })).toBeInTheDocument()
    expect(screen.queryByText('Web search')).toBeNull()
  })

  it("keeps the newest answer by checked_at, and Check again's when a time can't be read", () => {
    const older = readiness([CORE], '2026-10-30T07:00:00Z')
    const newer = readiness([SEARCH], '2026-10-30T09:00:00Z')
    const unread = readiness([CALENDAR], '')
    expect(newestReadiness(newer, older)).toBe(newer)
    expect(newestReadiness(older, newer)).toBe(newer)
    expect(newestReadiness(newer, unread)).toBe(unread)
    expect(newestReadiness(unread, older)).toBe(older)
    expect(newestReadiness(null, undefined)).toBeNull()
    expect(newestReadiness(undefined, older)).toBe(older)
  })

  it("says 'Checked a moment ago' on a rate-limited answer", () => {
    const limited = refusal(429, { code: 'checked_recently', detail: 'Checked a moment ago.' })
    hooks.check = { ...hooks.check, isError: true, error: limited }
    render(<ReadinessList readiness={readiness()} />)
    expect(screen.getByText(CHECKED_A_MOMENT_AGO)).toHaveAttribute('role', 'status')
    expect(screen.queryByRole('alert')).toBeNull()
    expect(screen.getByRole('button', { name: CHECK_AGAIN })).toBeEnabled()
  })

  it("gives another refusal one line: the server's sentence, or the list's own for a server error", () => {
    hooks.check = { ...hooks.check, isError: true, error: refusal(403, { code: 'forbidden', detail: 'You can’t check this workspace.' }) }
    const { rerender } = render(<ReadinessList readiness={readiness()} />)
    expect(screen.getByRole('alert')).toHaveTextContent('You can’t check this workspace.')
    hooks.check = { ...hooks.check, error: refusal(502, { detail: 'upstream db-1 timed out' }) }
    rerender(<ReadinessList readiness={readiness()} />)
    expect(screen.getByRole('alert')).toHaveTextContent(REFUSED_FALLBACK)
  })
})

const LOOKS = [
  { name: 'Studio light', studio: true, ground: 'studio' },
  { name: 'Studio dark', studio: true, ground: 'studio dark' },
  { name: 'Classic', studio: false, ground: '' },
] as const

describe.each(LOOKS)('ReadinessList in $name', ({ studio, ground }) => {
  it("draws every part from tokens, the state through the look's own status (Studio's pill names both ground tones)", () => {
    hooks.studio = studio
    const { container } = render(
      <div className={ground}>
        <ReadinessList readiness={readiness()} />
      </div>,
    )
    const pill = within(rowOf(container, 'calendar')).getByText('Needs setup').parentElement as HTMLElement
    expect(pill.classList.contains('studio-pill')).toBe(studio)
    if (studio) expect(pill.className).toContain('text-[hsl(var(--info))] dark:text-[hsl(var(--info))]')
    expect(screen.getByText(READINESS_TITLE)).toHaveClass('studio-eyebrow')
    expect(screen.getByRole('link', { name: 'Connect calendar' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: CHECK_AGAIN })).toBeInTheDocument()
  })
})

describe('readiness list source', () => {
  it('uses tokens only: no hex, no orange, no rounded-3xl', () => {
    const source = proactiveSource('readiness-list.tsx')
    expect(source).not.toMatch(HEX_COLOUR)
    expect(source).not.toMatch(ORANGE_CLASS)
    expect(source).not.toMatch(ROUNDED_3XL)
  })

  it(`keeps each component at most ${MAX_COMPONENT_LINES} lines`, () => {
    const spans = componentSpans(proactiveSource('readiness-list.tsx'))
    expect(spans.map((span) => span.name)).toEqual(expect.arrayContaining(['ReadinessList', 'ReadinessRow', 'CheckAgain']))
    spans.forEach((span) => expect(span.lines, span.name).toBeLessThanOrEqual(MAX_COMPONENT_LINES))
  })
})
