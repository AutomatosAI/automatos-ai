/**
 * PRD-268 §6.12.6, US-K02 (SDK 1.4.0) — BriefVoice says which of the brief's rows Auto is reading: the root's
 * `data-voice-row` and `onSpeakingRow` carry the row's index (the number Auto says, less one) while its words play,
 * and null between rows, during the greeting and once the reading ends or stops (brief-voice-rows.ts).
 *
 * The row is read off the speech's own events: the server voice (each clip's start and end) and, after a refusal
 * code, the browser's speech (each utterance's), over the server-speech fakes, played by hand. The voice engine is
 * not changed: the same fakes drive brief-voice.test.tsx.
 */
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { SynthesisPort } from '@/lib/voice/quick-ask/speaker'

vi.mock('@/lib/api-client', () => ({ apiClient: {} }))

const speech = vi.hoisted(() => ({ make: null as unknown as () => SynthesisPort }))
const voice = vi.hoisted(() => ({ current: null as unknown }))
vi.mock('@/components/voice/voice-provider', () => ({ useOptionalHostVoice: () => voice.current }))
vi.mock('@/lib/voice/quick-ask/deps', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@/lib/voice/quick-ask/deps')>()),
  speechPort: () => speech.make(),
}))

import { hostVoiceStub } from '@/components/voice/__tests__/support/host-voice-stub'
import { BriefVoice, LISTEN_TO_THE_BRIEF } from '@/components/voice/brief-voice'
import { pieceRows, sentenceRows } from '@/components/voice/brief-voice-rows'
import { spokenSentences } from '@/components/voice/use-brief-voice'
import { STOP } from '@/components/voice/voice-controls'
import { flush } from '@/lib/voice/quick-ask/__tests__/quick-ask-harness'
import { serverSpeechSetup } from '@/lib/voice/quick-ask/__tests__/server-speech-fakes'

/** As the server writes it (services/brief/build.py spoken_words): the greeting, then each row's title numbered. */
const SPOKEN = 'Morning, Sam. 1. Lisbon flights. 2. The weekly report.'
/** A first row whose title is longer than one piece of speech (speechPieces cuts at 200 characters). */
const LONG_TITLE = Array.from({ length: 45 }, () => 'budget').join(' ')
const SPOKEN_LONG = `Morning. 1. ${LONG_TITLE}. 2. Payroll.`
const QUICK_LISTENING = { snapshot: { mode: 'quick' as const, phase: 'listening' as const, micOpen: true } }
const ROW_ATTR = 'data-voice-row'

type Setup = ReturnType<typeof serverSpeechSetup>
let setup: Setup

function draw(spoken: string = SPOKEN) {
  const onSpeakingRow = vi.fn()
  const view = render(<BriefVoice spoken={spoken} onSpeakingRow={onSpeakingRow} />)
  const root = view.container.querySelector<HTMLElement>('[data-brief-voice="root"]')
  if (!root) throw new Error('no brief voice drawn')
  return { root, view, onSpeakingRow, reported: () => onSpeakingRow.mock.calls.map(([row]) => row) }
}

const settle = () => act(async () => flush())

async function press(): Promise<void> {
  fireEvent.click(screen.getByRole('button', { name: LISTEN_TO_THE_BRIEF }))
  await settle()
}

/** The server voice's clip `index` starts playing. */
function starts(index: number): void {
  act(() => setup.audio.plays[index].events.onPlaying())
}

/** The server voice's clip `index` ends. */
async function ends(index: number): Promise<void> {
  act(() => setup.audio.plays[index].events.onEnded())
  await settle()
}

/** The server voice answers every piece asked for so far. */
async function answerAll(): Promise<void> {
  setup.server.calls.forEach((_, index) => setup.server.answer(index))
  await settle()
}

beforeEach(() => {
  vi.spyOn(console, 'warn').mockImplementation(() => undefined)
  setup = serverSpeechSetup()
  speech.make = vi.fn(() => setup.port)
  voice.current = null
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe('the rows in what Listen reads', () => {
  it('gives each sentence the row its number starts, none to the greeting, and the row before to the rest', () => {
    expect(sentenceRows(spokenSentences(SPOKEN))).toEqual([null, 0, 1])
    expect(sentenceRows(['Morning.', '1. Ship the plan.', 'Then rest.', '2. Payroll.'])).toEqual([null, 0, 0, 1])
    expect(sentenceRows(['Nothing new needs you, and nothing is scheduled today.'])).toEqual([null])
  })

  it('starts a row only at the next number, so a number inside a title starts nothing', () => {
    expect(sentenceRows(['Morning.', '1. Close Q3.', '2026. Plan the budget.', '2. Payroll.'])).toEqual([null, 0, 0, 1])
    expect(sentenceRows(['Morning.', '1. 2026.', 'Plan the budget.', '2. Payroll.'])).toEqual([null, 0, 0, 1])
    expect(sentenceRows(['Morning.', '3. Payroll.', '1. Invoices.'])).toEqual([null, null, 0])
    expect(sentenceRows(['10. Invoices.'])).toEqual([null])
  })

  it('counts on past a row whose title has nothing to say, its number left before the next one’s', () => {
    expect(sentenceRows(['Morning.', '1. Invoices.', '2. 3.', 'Payroll.', '4. Travel.'])).toEqual([null, 0, 2, 2, 3])
  })

  it('gives every piece of a long row that row', () => {
    const sentences = spokenSentences(SPOKEN_LONG)
    expect(sentences).toHaveLength(3)
    expect(pieceRows(sentences)).toEqual([null, 0, 0, 1])
  })
})

describe('BriefVoice: the row Auto reads, in the server voice', () => {
  it('is null through the greeting, each row’s index while its clip plays, and null between rows and at the end', async () => {
    const { root, reported } = draw()
    await press()
    expect(root).not.toHaveAttribute(ROW_ATTR)

    await answerAll()
    starts(0)
    expect(root).not.toHaveAttribute(ROW_ATTR)
    await ends(0)
    await answerAll()

    starts(1)
    expect(root).toHaveAttribute(ROW_ATTR, '0')
    expect(root).toHaveAttribute('data-voice-speaking', 'true')
    await ends(1)
    expect(root).not.toHaveAttribute(ROW_ATTR)
    await answerAll()

    starts(2)
    expect(root).toHaveAttribute(ROW_ATTR, '1')
    await ends(2)
    expect(root).not.toHaveAttribute(ROW_ATTR)
    expect(root).not.toHaveAttribute('data-voice-speaking')
    expect(reported()).toEqual([0, null, 1, null])
  })

  it('is null once Stop is pressed mid-row, and the stopped reading’s late events report nothing', async () => {
    const { root, reported } = draw()
    await press()
    await answerAll()
    starts(0)
    await ends(0)
    await answerAll()
    starts(1)
    expect(root).toHaveAttribute(ROW_ATTR, '0')

    fireEvent.click(screen.getByRole('button', { name: STOP }))
    expect(root).not.toHaveAttribute(ROW_ATTR)
    await ends(1)
    expect(reported()).toEqual([0, null])
  })

  it('is null when the tab’s voice session starts and stops the reading', async () => {
    const { root, view, onSpeakingRow, reported } = draw()
    await press()
    await answerAll()
    starts(0)
    await ends(0)
    await answerAll()
    starts(1)
    voice.current = hostVoiceStub(QUICK_LISTENING)
    view.rerender(<BriefVoice spoken={SPOKEN} onSpeakingRow={onSpeakingRow} />)
    expect(root).not.toHaveAttribute(ROW_ATTR)
    expect(reported()).toEqual([0, null])
  })

  it('tells the theme a row whose clip starts and ends at once, in one batch', async () => {
    const { reported } = draw()
    await press()
    await answerAll()
    starts(0)
    await ends(0)
    await answerAll()
    act(() => {
      setup.audio.plays[1].events.onPlaying()
      setup.audio.plays[1].events.onEnded()
    })
    expect(reported()).toEqual([0, null])
  })

  it('tells the theme null when it is taken off the page mid-row', async () => {
    const { view, reported } = draw()
    await press()
    await answerAll()
    starts(0)
    await ends(0)
    await answerAll()
    starts(1)
    view.unmount()
    expect(reported()).toEqual([0, null])
  })

  it('reads on when the theme’s handler throws: the failure is logged', async () => {
    const onSpeakingRow = vi.fn<(row: number | null) => void>(() => {
      throw new Error('theme bug')
    })
    render(<BriefVoice spoken={SPOKEN} onSpeakingRow={onSpeakingRow} />)
    await press()
    await answerAll()
    starts(0)
    await ends(0)
    await answerAll()
    starts(1)
    await ends(1)
    await answerAll()
    starts(2)
    expect(onSpeakingRow.mock.calls.map(([row]) => row)).toEqual([0, null, 1])
    expect(console.warn).toHaveBeenCalledWith('[voice]', 'onSpeakingRow failed', expect.any(Error))
  })
})

describe('BriefVoice: the row Auto reads, in the browser’s speech', () => {
  it('follows each utterance once the server refuses (tts_off), a long row keeping its index across its pieces', async () => {
    const { root, reported } = draw(SPOKEN_LONG)
    await press()
    setup.server.refuse(0, 'tts_off', 503)
    await settle()
    expect(setup.browser.spoken()).toHaveLength(4)

    act(() => setup.browser.start(0))
    expect(root).not.toHaveAttribute(ROW_ATTR)
    act(() => setup.browser.end(0))
    act(() => setup.browser.start(1))
    expect(root).toHaveAttribute(ROW_ATTR, '0')
    act(() => setup.browser.end(1))
    expect(root).toHaveAttribute(ROW_ATTR, '0')
    act(() => setup.browser.start(2))
    expect(root).toHaveAttribute(ROW_ATTR, '0')
    act(() => setup.browser.end(2))
    expect(root).not.toHaveAttribute(ROW_ATTR)
    act(() => setup.browser.start(3))
    expect(root).toHaveAttribute(ROW_ATTR, '1')
    act(() => setup.browser.end(3))
    expect(root).not.toHaveAttribute(ROW_ATTR)
    expect(reported()).toEqual([0, null, 1, null])
  })
})
