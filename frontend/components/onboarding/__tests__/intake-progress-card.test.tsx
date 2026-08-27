/**
 * PRD-222 US-015 — the intake progress card.
 *
 * The card reuses the wizard's `useWizardProgress` SSE hook (mocked here) and
 * renders the five pipeline stages plus the two terminal states: profile-ready
 * (a "here's what I learned" handoff) and failed / stream-error (an honest error
 * with an "upload docs instead" fallback). Pure/mocked — only the shared hook is
 * stubbed, so the wizard's own consumption of it is provably untouched.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'

const { useWizardProgressMock } = vi.hoisted(() => ({
  useWizardProgressMock: vi.fn(),
}))

vi.mock('@/hooks/use-wizard-progress', () => ({
  useWizardProgress: (opts: unknown) => useWizardProgressMock(opts),
}))

import { IntakeProgressCard } from '../intake-progress-card'

type Ev = { ts: number; stage: string; level: string; message: string; meta: Record<string, unknown> }

function ev(stage: string, message: string, level = 'info'): Ev {
  return { ts: 1_700_000_000 + stage.length, stage, level, message, meta: {} }
}

function mockProgress(state: string, events: Ev[]) {
  useWizardProgressMock.mockReturnValue({
    events,
    state,
    latest: events.length ? events[events.length - 1] : null,
    reset: vi.fn(),
  })
}

describe('IntakeProgressCard (US-015)', () => {
  beforeEach(() => useWizardProgressMock.mockReset())

  it('renders nothing without a profileId', () => {
    mockProgress('idle', [])
    const { container } = render(<IntakeProgressCard profileId={null} />)
    expect(screen.queryByTestId('intake-progress-card')).toBeNull()
    expect(container).toBeEmptyDOMElement()
  })

  it('renders the five pipeline stages from the shared hook', () => {
    mockProgress('streaming', [ev('scan', 'Mapping the site…')])
    render(<IntakeProgressCard profileId="p1" />)

    expect(screen.getByTestId('intake-progress-card')).toBeInTheDocument()
    for (const stage of ['scan', 'scrape', 'ingest', 'graphify', 'profile']) {
      expect(screen.getByTestId(`intake-stage-${stage}`)).toBeInTheDocument()
    }
    // PRD wording: graphify is labelled "Graph"
    expect(screen.getByTestId('intake-stage-graphify')).toHaveTextContent('Graph')
  })

  it('marks the current stage active and shows the latest status line while streaming', () => {
    mockProgress('streaming', [
      ev('scan', 'Mapped 12 pages'),
      ev('scrape', 'Scraping selected pages…'),
    ])
    render(<IntakeProgressCard profileId="p1" />)

    expect(screen.getByTestId('intake-stage-scrape')).toHaveAttribute('data-status', 'active')
    // an earlier stage is marked done
    expect(screen.getByTestId('intake-stage-scan')).toHaveAttribute('data-status', 'done')
    expect(screen.getByTestId('intake-status')).toHaveTextContent('Scraping selected pages…')
    // no terminal copy yet
    expect(screen.queryByTestId('intake-handoff')).toBeNull()
    expect(screen.queryByTestId('intake-error')).toBeNull()
  })

  it('terminal success renders the "here\'s what I learned" handoff', () => {
    mockProgress('complete', [
      ev('scan', 'done'),
      ev('profile', 'Profile ready'),
    ])
    render(<IntakeProgressCard profileId="p1" />)

    expect(screen.getByTestId('intake-progress-card')).toHaveAttribute('data-state', 'complete')
    expect(screen.getByTestId('intake-handoff')).toHaveTextContent(/here.?s what i learned/i)
    // every stage reads done on completion
    expect(screen.getByTestId('intake-stage-profile')).toHaveAttribute('data-status', 'done')
    expect(screen.queryByTestId('intake-error')).toBeNull()
  })

  it('terminal failure renders the honest error + upload-docs fallback', () => {
    const onUploadDocs = vi.fn()
    mockProgress('failed', [ev('scrape', 'Firecrawl returned 503', 'error')])
    render(<IntakeProgressCard profileId="p1" onUploadDocs={onUploadDocs} />)

    expect(screen.getByTestId('intake-error')).toHaveTextContent('Firecrawl returned 503')
    const fallback = screen.getByTestId('intake-upload-fallback')
    expect(fallback).toHaveTextContent(/upload your docs instead/i)
    fireEvent.click(fallback)
    expect(onUploadDocs).toHaveBeenCalled()
    expect(screen.queryByTestId('intake-handoff')).toBeNull()
  })

  it('a dropped stream (state=error) is treated as a failure with the same fallback', () => {
    mockProgress('error', [ev('ingest', 'connecting…')])
    render(<IntakeProgressCard profileId="p1" />)

    expect(screen.getByTestId('intake-progress-card')).toHaveAttribute('data-state', 'error')
    expect(screen.getByTestId('intake-upload-fallback')).toBeInTheDocument()
    // honest fallback copy when the stream itself dropped
    expect(screen.getByTestId('intake-error')).toBeInTheDocument()
  })
})
