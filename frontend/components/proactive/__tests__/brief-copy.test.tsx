/**
 * PRD-268 W3 US-C23 (§6.11, K16) — the default BriefCard draws the server's copy: each section's label and the
 * empty sentence from `brief.copy` (in the person's language), else from `brief.labels` (a brief from before W3, or
 * a theme's: the SDK's Brief carries labels only, K17). "Show N more" stays the server's own line from
 * `labels.more`: a viewer's `counts.needs_you` is the badge's total, so the card never fills the template from
 * counts. The wire keeps only a known section's label and the copy's text fields.
 */
import { render, screen, within } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

vi.mock('@/hooks/use-digest-api', () => ({
  useSubmitDigestFeedback: () => ({ mutate: vi.fn() }),
  // SDK 1.4.0 (#1383): the card's thumbs read the person's remembered rating; none here.
  useDigestRating: () => ({ data: undefined }),
}))

import { BriefCard } from '@/components/proactive/brief-card'
import { checkedBrief } from '@/lib/theme-sdk/host-cards/proactive-props'
import { briefOf } from '@/lib/proactive/wire'
import { BRIEF_SECTIONS, type Brief, type BriefCopy, type BriefRow } from '@/lib/proactive/types'

const PORTUGUESE: BriefCopy = {
  section_labels: {
    needs_you: 'Precisa de você',
    suggestions: 'Sugestões',
    today: 'Hoje',
    since: 'Desde o seu último resumo',
    unchecked: 'Não deu para verificar',
  },
  empty_sentence: 'Nada novo precisa de você, e nada está agendado para hoje.',
  show_more_template: 'Mostrar mais {n}',
}
const ENGLISH_LABELS = { needs_you: 'Needs you', suggestions: 'Suggested', today: 'Today', since: 'Since your last brief', unchecked: "Couldn't check" }
const NO_ROWS = { needs_you: [], suggestions: [], today: [], since: [], unchecked: [] }
const NO_COUNTS = { needs_you: 0, suggestions: 0, today: 0, since: 0 }

const row = (id: string): BriefRow => ({ id, title: `Title ${id}`, why: `Why ${id}.` })
const rows = (prefix: string, many: number): BriefRow[] => Array.from({ length: many }, (_, n) => row(`${prefix}${n}`))

function brief(overrides: Partial<Brief> = {}): Brief {
  return {
    v: 1,
    date: '2026-10-30',
    greeting: 'Bom dia, Ana.',
    prose: 'Duas coisas mudaram.',
    spoken: 'Bom dia, Ana.',
    sections: { needs_you: rows('n', 2), suggestions: rows('s', 1), today: rows('t', 1), since: rows('d', 1), unchecked: rows('u', 1) },
    counts: { needs_you: 2, suggestions: 1, today: 1, since: 1 },
    labels: { sections: ENGLISH_LABELS, more: {}, empty: null },
    copy: PORTUGUESE,
    state_hash: 'hash-1',
    generated_at: '2026-10-30T07:00:00Z',
    ...overrides,
  }
}

const sectionOf = (container: HTMLElement, name: string) => container.querySelector(`section[data-section="${name}"]`) as HTMLElement

describe("BriefCard draws the server's copy", () => {
  it("names every section with the copy's label, in the person's language", () => {
    const { container } = render(<BriefCard brief={brief()} />)
    for (const section of BRIEF_SECTIONS) {
      const label = PORTUGUESE.section_labels[section] as string
      expect(sectionOf(container, section)).toHaveAttribute('aria-label', label)
      expect(within(sectionOf(container, section)).getByText(label, { exact: false })).toBeInTheDocument()
    }
    expect(screen.queryByText(ENGLISH_LABELS.needs_you, { exact: false })).toBeNull()
  })

  it("says the copy's empty sentence when no section has a row", () => {
    const { container } = render(<BriefCard brief={brief({ sections: NO_ROWS, counts: NO_COUNTS })} />)
    expect(screen.getByText(PORTUGUESE.empty_sentence as string)).toHaveAttribute('data-ai-generated', 'true')
    expect(container.querySelector('section[data-section]')).toBeNull()
  })

  it("keeps the server's own Show N more line and never fills the template from the counts", () => {
    // A viewer: the badge counts 8, the fold counted only the 3 rows they may see, so the server folded nothing.
    const viewer = brief({ sections: { ...NO_ROWS, needs_you: rows('n', 3) }, counts: { ...NO_COUNTS, needs_you: 8 } })
    const { rerender } = render(<BriefCard brief={viewer} />)
    expect(screen.queryByText(/Mostrar mais/)).toBeNull()

    const folded = brief({
      sections: { ...NO_ROWS, needs_you: rows('n', 5) },
      counts: { ...NO_COUNTS, needs_you: 7 },
      labels: { sections: ENGLISH_LABELS, more: { needs_you: 'Mostrar mais 2' }, empty: null },
    })
    rerender(<BriefCard brief={folded} />)
    expect(screen.getByText('Mostrar mais 2')).toBeInTheDocument()
  })

  it('reads the labels when a brief carries no copy (built before W3, or handed over by a theme)', () => {
    const older = brief({ copy: undefined, sections: NO_ROWS, counts: NO_COUNTS, labels: { sections: ENGLISH_LABELS, more: {}, empty: 'Nothing new needs you.' } })
    render(<BriefCard brief={older} />)
    expect(screen.getByText('Nothing new needs you.')).toBeInTheDocument()

    const { container } = render(<BriefCard brief={brief({ copy: undefined })} />)
    expect(sectionOf(container, 'today')).toHaveAttribute('aria-label', 'Today')
  })
})

describe("the wire reads the brief's copy", () => {
  const raw = { ...brief(), copy: undefined }

  it('keeps a known section label and the text fields, and nothing else', () => {
    const read = briefOf({
      ...raw,
      copy: { section_labels: { needs_you: 'Precisa de você', bogus: 'x', today: 7 }, empty_sentence: 'Nada.', show_more_template: 'Mostrar mais {n}', later: 1 },
    })
    expect(read?.copy).toEqual({ section_labels: { needs_you: 'Precisa de você' }, empty_sentence: 'Nada.', show_more_template: 'Mostrar mais {n}' })
  })

  it('reads no copy when the brief has none, or when it is not an object', () => {
    expect(briefOf(raw)?.copy).toBeUndefined()
    expect(briefOf({ ...raw, copy: 'Needs you' })?.copy).toBeUndefined()
    expect(briefOf({ ...raw, copy: { section_labels: null, empty_sentence: 3 } })?.copy).toEqual({
      section_labels: {},
      empty_sentence: undefined,
      show_more_template: undefined,
    })
  })

  it("keeps the copy of a brief a theme hands the SDK's BriefCard (the host's brief at run time)", () => {
    expect(checkedBrief({ ...raw, copy: PORTUGUESE }).copy).toEqual(PORTUGUESE)
  })
})
