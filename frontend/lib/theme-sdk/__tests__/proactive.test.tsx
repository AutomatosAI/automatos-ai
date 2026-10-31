/**
 * PRD-268 US-C21e (SDK 1.3.0) — the proactive area, the default components and BriefVoice reach a theme through
 * '@automatos/theme-sdk' and behave as the host's own.
 *
 * - The SDK module hands a theme the same functions as the area modules, and DISPLAY_STATE_LABELS is the host's table.
 * - Each hook asks the host's endpoint into the host's cache (a theme's read and the host's surfaces share one
 *   answer), hands back a frozen copy, and checks a theme's input before anything is sent; while the server offers
 *   no calibration (CALIBRATION_ENABLED ships off), calibration reads the host's answer and sends nothing.
 * - Each default component is the host's (its words, its names, its protected surface, its route), with the theme's
 *   props checked: a prop of the wrong kind throws, naming the card and the prop, and what a card hands back is frozen.
 */
import { act, cleanup, fireEvent, render, renderHook, screen, waitFor, within } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import React, { type ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const requestMock = vi.hoisted(() => vi.fn())
const postMock = vi.hoisted(() => vi.fn())
vi.mock('@/lib/api-client', () => ({
  apiClient: {
    request: (...args: unknown[]) => requestMock(...args),
    post: (...args: unknown[]) => postMock(...args),
  },
}))
vi.mock('@/components/workspace-provider', () => ({ useWorkspaceOptional: () => ({ workspace: { id: 'ws-1' } }) }))
vi.mock('@/hooks/use-digest-api', () => ({ useSubmitDigestFeedback: () => ({ mutate: vi.fn() }) }))
vi.mock('next/link', () => ({
  default: ({ href, children, ...rest }: { href: string; children: ReactNode; className?: string }) => (
    <a href={href} {...rest}>
      {children}
    </a>
  ),
}))

import * as sdk from '@automatos/theme-sdk'
import { PROTECTED_SURFACE_ATTR } from '@/components/theme-host/protected-surface'
import { SOUNDS_RIGHT } from '@/components/proactive/calibration-card'
import { REFUSED_FALLBACK } from '@/components/proactive/use-suggestion-card'
import { LISTEN_TO_THE_BRIEF } from '@/components/voice/brief-voice'
import { briefQueryKeys } from '@/hooks/use-brief'
import { CALIBRATION_PATH, CALIBRATION_UNAVAILABLE } from '@/hooks/use-calibration'
import { readinessQueryKeys } from '@/hooks/use-readiness'
import { suggestionQueryKeys } from '@/hooks/use-suggestions'
import { localDateOf } from '@/lib/proactive/brief-day'
import { DISPLAY_STATE_LABELS } from '@/lib/proactive/display-state'

import * as cardsArea from '../cards'
import * as proactiveArea from '../proactive'
import * as themeArea from '../theme'
import * as voiceArea from '../voice'

const ID = '7d1c0a52-3e0b-4c55-9a1e-2f6f0c1d9b10'
const CHAT = '5b7f2c1e-8d4a-4f6b-9c3e-1a2b3c4d5e6f'
const MESSAGE = '9e8d7c6b-5a4f-4e3d-8c2b-1a0f9e8d7c6b'
const TITLE = 'Look at the Lisbon flights'
const SUGGESTIONS_PATH = '/api/proactive/suggestions?status=open'
const READINESS_PATH = '/api/readiness'
const TODAY = localDateOf(new Date())

const SUGGESTION = {
  id: ID, workspace_id: 'ws-1', user_id: 1, source_type: 'check_in', source_agent_id: 4, run_id: null, title: TITLE,
  detail: 'Three runs failed since Monday.', evidence: [{ type: 'ticket', id: '0419' }],
  proposed_action: { kind: 'board_task', payload: {} }, subject_key: 'ticket:0419', severity: 'today', status: 'open',
  occurrences: 1, last_seen_at: null, started_ref: null, decided_via: null, hide_reason: null, decided_at: null,
  est_credits: null, created_at: null, delivered_at: null, expires_at: null,
} as const satisfies sdk.Suggestion

const ROW = { id: 'sugg:1', title: 'Lisbon flights', why: 'Prices dropped', action: { verb: 'start', label: 'Start' } } as const
const BRIEF = {
  v: 1, date: TODAY, greeting: 'Morning, Sam.', prose: 'Two things need you today.',
  spoken: 'Morning, Sam. 1. Lisbon flights.',
  sections: { needs_you: [], suggestions: [ROW], today: [], since: [], unchecked: [] },
  counts: { needs_you: 0, suggestions: 1, today: 0, since: 0 },
  labels: { sections: { suggestions: 'Suggestions' }, more: {}, empty: 'Nothing needs you.' },
  state_hash: 'h1', generated_at: '2026-10-30T07:00:00Z',
} as const satisfies sdk.Brief

const READINESS = {
  v: 1, checked_at: '2026-10-30T07:00:00Z',
  items: [{ id: 'calendar', label: 'Calendar', detail: 'Not connected', state: 'needs_setup', fix: { label: 'Connect calendar', href: '/settings/integrations' }, needed_by: ['brief'] }],
} as const satisfies sdk.Readiness

const CALIBRATION = {
  v: 1, id: 'cal-1', empty: false, offers: ['Draft the weekly report', 'Plan a trip', 'Tidy the board'],
  lines: [{ id: 'l1', text: 'You run a travel agency.', sources: [{ type: 'deliverable', id: 'd1', label: 'Q3 plan' }] }],
} as const satisfies sdk.Calibration

function setup() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  const wrapper = ({ children }: { children: ReactNode }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>
  return { client, wrapper }
}

function draw(element: React.ReactElement) {
  return render(element, { wrapper: setup().wrapper })
}

function silenceBoundary() {
  vi.spyOn(console, 'error').mockImplementation(() => undefined)
}

beforeEach(() => {
  requestMock.mockReset()
  postMock.mockReset()
  requestMock.mockImplementation(async (path: string) => {
    if (path === SUGGESTIONS_PATH) return { suggestions: [SUGGESTION] }
    if (path === READINESS_PATH) return READINESS
    if (path.startsWith(CALIBRATION_PATH)) return { available: false, offer: false, consent_line: '', status: null, calibration: null, rejected_lines: [], accepted_lines: [] }
    if (path === `/api/chat/${CHAT}/messages`) return [{ id: MESSAGE, parts: [{ type: 'text', text: BRIEF.prose }, { type: 'brief@1', brief: BRIEF }] }]
    throw new Error(`unexpected request ${path}`)
  })
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe("'@automatos/theme-sdk' — the proactive names", () => {
  it('hands a theme the area modules’ own hooks, cards and BriefVoice, and the host’s DISPLAY_STATE_LABELS', () => {
    const hooks = ['useBrief', 'useReadiness', 'useCheckReadiness', 'useSuggestions', 'useStartSuggestion', 'useHideSuggestion', 'useCalibration', 'useAnswerCalibration'] as const
    hooks.forEach((name) => expect(sdk[name], name).toBe(proactiveArea[name]))
    const cards = ['BriefCard', 'SuggestionCard', 'StatusLabel', 'ReadinessList', 'CalibrationCard', 'ReplyChips'] as const
    cards.forEach((name) => expect(sdk[name], name).toBe(cardsArea[name]))
    expect(sdk.BriefVoice).toBe(voiceArea.BriefVoice)
    expect(sdk.DISPLAY_STATE_LABELS).toBe(themeArea.DISPLAY_STATE_LABELS)
    expect(sdk.DISPLAY_STATE_LABELS).toBe(DISPLAY_STATE_LABELS)
    expect(sdk.DISPLAY_STATE_LABELS.needs_you).toMatchObject({ glyph: '◆', word: 'Needs you' })
  })
})

describe('proactive hooks — the host’s endpoints and cache', () => {
  it('useSuggestions reads the host’s list into the host’s cache and hands back a frozen copy', async () => {
    const { client, wrapper } = setup()
    const { result } = renderHook(() => sdk.useSuggestions(), { wrapper })
    await waitFor(() => expect(result.current.data).toHaveLength(1))
    expect(requestMock).toHaveBeenCalledWith(SUGGESTIONS_PATH)
    expect(client.getQueryData(suggestionQueryKeys.open)).toEqual([expect.objectContaining({ id: ID, title: TITLE })])
    expect(Object.isFrozen(result.current)).toBe(true)
    expect(Object.isFrozen(result.current.data)).toBe(true)
    expect(() => (result.current.data as sdk.Suggestion[]).push(SUGGESTION)).toThrow(TypeError)
    expect(result.current.data?.[0]).not.toBe((client.getQueryData(suggestionQueryKeys.open) as unknown[])[0])
  })

  it('useStartSuggestion posts the host’s Start route and answers a frozen copy', async () => {
    postMock.mockResolvedValue({ suggestion: { ...SUGGESTION, status: 'started' }, opens: { type: 'board_task', board_task_id: 7 } })
    const { result } = renderHook(() => sdk.useStartSuggestion(), { wrapper: setup().wrapper })
    let answer: sdk.StartAnswer | undefined
    await act(async () => {
      answer = await result.current.runAsync({ id: ID })
    })
    expect(postMock).toHaveBeenCalledWith(`/api/proactive/suggestions/${ID}/start`, {})
    expect(answer?.opens).toEqual({ type: 'board_task', board_task_id: 7 })
    expect(Object.isFrozen(answer)).toBe(true)
  })

  it('useStartSuggestion and useHideSuggestion throw on an id that isn’t a suggestion’s, before anything is sent', () => {
    const { wrapper } = setup()
    const start = renderHook(() => sdk.useStartSuggestion(), { wrapper }).result.current
    const hide = renderHook(() => sdk.useHideSuggestion(), { wrapper }).result.current
    expect(() => start.run({ id: '../admin' })).toThrow(/^Automatos theme SDK: useStartSuggestion needs the id/)
    expect(() => hide.run({ id: '42' })).toThrow(/^Automatos theme SDK: useHideSuggestion needs the id/)
    expect(postMock).not.toHaveBeenCalled()
  })

  it('useHideSuggestion posts the reason, and refuses one Hide doesn’t know', async () => {
    postMock.mockResolvedValue({ suggestion: { ...SUGGESTION, status: 'hidden' } })
    const { result } = renderHook(() => sdk.useHideSuggestion(), { wrapper: setup().wrapper })
    expect(() => result.current.run({ id: ID, reason: 'spam' as sdk.HideReason })).toThrow(/needs reason to be one of not_now/)
    expect(postMock).not.toHaveBeenCalled()
    await act(async () => {
      await result.current.runAsync({ id: ID, reason: 'not_useful' })
    })
    expect(postMock).toHaveBeenCalledWith(`/api/proactive/suggestions/${ID}/hide`, { reason: 'not_useful' })
  })

  it('useReadiness and useCheckReadiness share the host’s readiness cache', async () => {
    const { client, wrapper } = setup()
    const { result } = renderHook(() => ({ read: sdk.useReadiness(), check: sdk.useCheckReadiness() }), { wrapper })
    await waitFor(() => expect(result.current.read.data?.items).toHaveLength(1))
    expect(requestMock).toHaveBeenCalledWith(READINESS_PATH)
    postMock.mockResolvedValue({ ...READINESS, checked_at: '2026-10-30T08:00:00Z' })
    await act(async () => {
      await result.current.check.runAsync()
    })
    expect(postMock).toHaveBeenCalledWith(`${READINESS_PATH}/check`)
    expect(client.getQueryData(readinessQueryKeys.all)).toMatchObject({ checked_at: '2026-10-30T08:00:00Z' })
    await waitFor(() => expect(result.current.read.data?.checked_at).toBe('2026-10-30T08:00:00Z'))
  })

  it('useBrief reads the brief the host opened, and asks nothing before', async () => {
    const { client, wrapper } = setup()
    const { result } = renderHook(() => sdk.useBrief(), { wrapper })
    expect(result.current.data).toBeNull()
    expect(requestMock).not.toHaveBeenCalled()
    act(() => {
      client.setQueryData(briefQueryKeys.where('ws-1', TODAY), { chat_id: CHAT, message_id: MESSAGE })
    })
    await waitFor(() => expect(result.current.data?.greeting).toBe(BRIEF.greeting))
    expect(requestMock).toHaveBeenCalledWith(`/api/chat/${CHAT}/messages`)
    expect(Object.isFrozen(result.current.data)).toBe(true)
  })

  it('useCalibration and useAnswerCalibration while the server offers none: nothing offered, an answer refused, nothing sent', async () => {
    const { result } = renderHook(() => ({ read: sdk.useCalibration(), answer: sdk.useAnswerCalibration() }), { wrapper: setup().wrapper })
    await waitFor(() => expect(result.current.read.isLoading).toBe(false))
    expect(result.current.read).toMatchObject({ data: null, available: false, isLoading: false, error: null })
    expect(result.current.answer.available).toBe(false)
    await expect(result.current.answer.runAsync({ calibration_id: 'cal-1', answer: 'accept' })).rejects.toThrow(CALIBRATION_UNAVAILABLE)
    act(() => result.current.answer.run({ calibration_id: 'cal-1', answer: 'reject', line_id: 'l1' }))
    await waitFor(() => expect(result.current.answer.error).toBe(CALIBRATION_UNAVAILABLE))
    expect(() => result.current.answer.run({ calibration_id: '', answer: 'accept' })).toThrow(/^Automatos theme SDK: useAnswerCalibration/)
    expect(requestMock.mock.calls.every(([path]) => String(path).startsWith(CALIBRATION_PATH))).toBe(true)
    expect(postMock).not.toHaveBeenCalled()
  })
})

describe('the default components — the host’s own, the theme’s props checked', () => {
  it('BriefCard is the host’s card with the SDK’s Listen, and hands a row’s action back as a frozen copy', async () => {
    const onAction = vi.fn()
    draw(<sdk.BriefCard brief={BRIEF} onAction={onAction} />)
    const card = screen.getByRole('article', { name: BRIEF.greeting })
    expect(within(card).getByText(BRIEF.greeting)).toHaveAttribute('data-ai-generated', 'true')
    const listen = within(card).getByRole('button', { name: LISTEN_TO_THE_BRIEF })
    expect(listen.closest(`[${PROTECTED_SURFACE_ATTR}="brief-voice"]`)).not.toBeNull()
    fireEvent.click(within(card).getByRole('button', { name: `Start: ${ROW.title}` }))
    await waitFor(() => expect(onAction).toHaveBeenCalledWith(expect.objectContaining({ id: ROW.id }), 'suggestions'))
    expect(Object.isFrozen(onAction.mock.calls[0][0])).toBe(true)
  })

  it('BriefCard leaves Listen out when the theme places BriefVoice itself, and refuses a non-brief', () => {
    draw(<sdk.BriefCard brief={BRIEF} listen={false} />)
    expect(screen.queryByRole('button', { name: LISTEN_TO_THE_BRIEF })).toBeNull()
    silenceBoundary()
    expect(() => draw(<sdk.BriefCard brief={{ v: 2 } as unknown as sdk.Brief} />)).toThrow(/^Automatos theme SDK: BriefCard needs brief /)
    expect(() => draw(<sdk.BriefCard brief={BRIEF} onOpen={'go' as never} />)).toThrow(/BriefCard needs onOpen to be a function/)
  })

  it('SuggestionCard is the host’s card: its Start posts the host’s route and hands the theme a frozen answer', async () => {
    postMock.mockResolvedValue({ suggestion: { ...SUGGESTION, status: 'started' }, opens: { type: 'chat', suggestion_id: ID } })
    const onStarted = vi.fn()
    draw(<sdk.SuggestionCard suggestion={SUGGESTION} onStarted={onStarted} />)
    expect(screen.getByRole('article', { name: `Auto suggests: ${TITLE}` })).toBeInTheDocument()
    expect(screen.getByText(TITLE).closest('[data-ai-generated="true"]')).not.toBeNull()
    fireEvent.click(screen.getByRole('button', { name: `Start: ${TITLE}` }))
    await waitFor(() => expect(onStarted).toHaveBeenCalledTimes(1))
    expect(postMock).toHaveBeenCalledWith(`/api/proactive/suggestions/${ID}/start`, { via: 'web' })
    expect(Object.isFrozen(onStarted.mock.calls[0][0])).toBe(true)
  })

  it('SuggestionCard refuses a suggestion whose id isn’t one, and an unknown variant, before anything is sent', () => {
    silenceBoundary()
    expect(() => draw(<sdk.SuggestionCard suggestion={{ ...SUGGESTION, id: '../../x' }} />)).toThrow(/^Automatos theme SDK: SuggestionCard needs suggestion /)
    expect(() => draw(<sdk.SuggestionCard suggestion={SUGGESTION} variant={'banner' as never} />)).toThrow(/SuggestionCard needs variant to be one of card, pinned/)
    expect(postMock).not.toHaveBeenCalled()
  })

  it('StatusLabel draws the host’s glyph and word, and refuses a retry it doesn’t know', () => {
    draw(<sdk.StatusLabel state="needs_you" nextStep="Waiting for your approval" />)
    expect(screen.getByText('Needs you')).toBeInTheDocument()
    expect(screen.getByText('Waiting for your approval')).toBeInTheDocument()
    silenceBoundary()
    expect(() => draw(<sdk.StatusLabel state="failed" retry={'reboot' as never} />)).toThrow(/^Automatos theme SDK: StatusLabel needs retry /)
  })

  it('ReadinessList is the host’s list, and a read’s failure is the host’s own line, never the theme’s text', () => {
    draw(<sdk.ReadinessList readiness={READINESS} />)
    expect(screen.getByRole('listitem', { name: 'Calendar: needs setup. Connect calendar.' })).toBeInTheDocument()
    cleanup()
    draw(<sdk.ReadinessList readiness={null} error="<b>theme text</b>" />)
    expect(screen.getByRole('alert')).toHaveTextContent(REFUSED_FALLBACK)
    expect(screen.queryByText(/theme text/)).toBeNull()
  })

  it('CalibrationCard renders from its props and hands Sounds right back as a frozen answer', async () => {
    const onAnswer = vi.fn().mockResolvedValue(undefined)
    draw(<sdk.CalibrationCard calibration={CALIBRATION} onAnswer={onAnswer} />)
    expect(screen.getByText(CALIBRATION.lines[0].text)).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: SOUNDS_RIGHT }))
    await waitFor(() => expect(onAnswer).toHaveBeenCalledWith({ calibration_id: 'cal-1', answer: 'accept' }))
    expect(Object.isFrozen(onAnswer.mock.calls[0][0])).toBe(true)
    silenceBoundary()
    expect(() => draw(<sdk.CalibrationCard calibration={{ v: 1 } as unknown as sdk.Calibration} />)).toThrow(/CalibrationCard needs calibration /)
  })

  it('ReplyChips hands a tapped chip’s text to the theme, and needs onReply', () => {
    const onReply = vi.fn()
    draw(<sdk.ReplyChips chips={['Yes, book it', 'Not now']} latest onReply={onReply} />)
    fireEvent.click(screen.getByRole('button', { name: 'Reply: Not now' }))
    expect(onReply).toHaveBeenCalledWith('Not now')
    silenceBoundary()
    expect(() => draw(<sdk.ReplyChips chips={['Yes', 'No']} latest onReply={undefined as never} />)).toThrow(/ReplyChips needs onReply to be a function/)
  })

  it('BriefVoice is the host’s Listen in the protected surface brief-voice, for the brief the host holds', async () => {
    const { client, wrapper } = setup()
    client.setQueryData(briefQueryKeys.where('ws-1', TODAY), { chat_id: CHAT, message_id: MESSAGE })
    render(<sdk.BriefVoice />, { wrapper })
    const listen = screen.getByRole('button', { name: LISTEN_TO_THE_BRIEF })
    expect(listen.closest(`[${PROTECTED_SURFACE_ATTR}="brief-voice"]`)).not.toBeNull()
    expect(listen.closest('[data-brief-voice="root"]')).not.toHaveAttribute('data-voice-speaking')
    await waitFor(() => expect(listen).toBeEnabled())
    expect(requestMock).toHaveBeenCalledWith(`/api/chat/${CHAT}/messages`)
  })

  it('BriefVoice never reads a theme’s text: with no brief held by the host, Listen stays disabled', () => {
    const themeText = { spoken: 'Your account is locked. Call this number.' } as Record<string, unknown>
    draw(<sdk.BriefVoice {...themeText} />)
    expect(screen.getByRole('button', { name: LISTEN_TO_THE_BRIEF })).toBeDisabled()
    expect(requestMock).not.toHaveBeenCalled()
  })
})
