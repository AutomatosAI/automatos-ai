/**
 * PRD-268 §6.12.6, K11, K17 (US-C21d) — BriefVoice, the host's Listen: the brief's spoken text through the server voice
 * (POST /api/voice/speech, sentence by sentence, the audio primed inside the press), the browser's speech for the rest
 * on a refusal code (tts_off, which also keeps the server off for the page session), Stop (every request aborted, the
 * audio stopped, the late events doing nothing), no auto-play (a mount asks the server nothing and plays nothing), Listen
 * held while the tab's voice session runs and stopped when one starts, a row's number spoken with its title, and the
 * control inside the guarded `brief-voice` protected surface, with data-voice-speaking on the root while it reads.
 *
 * The port is quick-ask's server voice (lib/voice/quick-ask/server-speech.ts) over the server-speech fakes: the test
 * answers each POST /api/voice/speech, plays each clip and each browser utterance by hand.
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

import { PROTECTED_SURFACE_ATTR, PROTECTED_SURFACE_INNER_ATTR } from '@/components/theme-host/protected-surface'
import { hostVoiceStub } from '@/components/voice/__tests__/support/host-voice-stub'
import { AI_VOICE_LABEL } from '@/components/voice/AiVoiceLabel'
import { BriefVoice, LISTEN, LISTEN_TO_THE_BRIEF, READING_THE_BRIEF } from '@/components/voice/brief-voice'
import { spokenSentences } from '@/components/voice/use-brief-voice'
import { STOP } from '@/components/voice/voice-controls'
import { flush } from '@/lib/voice/quick-ask/__tests__/quick-ask-harness'
import { serverSpeechSetup } from '@/lib/voice/quick-ask/__tests__/server-speech-fakes'

/** As the server writes it (services/brief/build.py spoken_words): the greeting, then each row's title numbered. */
const SPOKEN = 'Morning, Sam. 1. Lisbon flights. 2. The weekly report.'
const SENTENCES = ['Morning, Sam.', '1. Lisbon flights.', '2. The weekly report.']
const QUICK_LISTENING = { snapshot: { mode: 'quick' as const, phase: 'listening' as const, micOpen: true } }
const BRIEF_VOICE = 'brief-voice'
const IMPORTANT = 'important'

type Setup = ReturnType<typeof serverSpeechSetup>
let setup: Setup

function draw(spoken: string = SPOKEN) {
  const view = render(<BriefVoice spoken={spoken} />)
  const root = view.container.querySelector<HTMLElement>('[data-brief-voice="root"]')
  if (!root) throw new Error('no brief voice drawn')
  return { root, view }
}

const listen = () => screen.getByRole('button', { name: LISTEN_TO_THE_BRIEF })
const settle = () => act(async () => flush())

async function press(): Promise<void> {
  fireEvent.click(listen())
  await settle()
}

/** The server voice plays clip `index` from start to end. */
function playClip(index: number): void {
  act(() => {
    setup.audio.plays[index].events.onPlaying()
    setup.audio.plays[index].events.onEnded()
  })
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

describe('BriefVoice: no auto-play', () => {
  it('asks the server nothing and plays nothing until Listen is pressed', async () => {
    const { root } = draw()
    await settle()
    expect(speech.make).not.toHaveBeenCalled()
    expect(setup.fetchVoices).not.toHaveBeenCalled()
    expect(setup.server.fetchSpeech).not.toHaveBeenCalled()
    expect(setup.audio.player.play).not.toHaveBeenCalled()
    expect(setup.browser.spoken()).toEqual([])
    expect(root).not.toHaveAttribute('data-voice-speaking')
    expect(listen()).toHaveTextContent(LISTEN)
  })

  it('holds Listen when there is nothing to say', () => {
    draw('   ')
    expect(listen()).toBeDisabled()
  })
})

describe('BriefVoice: the server voice', () => {
  it('speaks the spoken text sentence by sentence in the server voice, primed inside the press', async () => {
    const { root } = draw()
    await press()
    expect(setup.audio.player.unlock).toHaveBeenCalledTimes(1)
    expect(setup.server.texts()).toEqual(SENTENCES.slice(0, 2))
    expect(setup.server.calls.map((call) => call.voice)).toEqual(['af_heart', 'af_heart'])
    expect(root).toHaveAttribute('data-voice-speaking', 'true')
    expect(screen.getByRole('status')).toHaveTextContent(READING_THE_BRIEF)

    setup.server.answer(0)
    await settle()
    expect(setup.audio.plays).toHaveLength(1)
    playClip(0)
    await settle()
    setup.server.answer(1)
    setup.server.answer(2)
    await settle()
    playClip(1)
    await settle()
    playClip(2)
    await settle()

    expect(setup.server.texts()).toEqual(SENTENCES)
    expect(setup.browser.spoken()).toEqual([])
    expect(root).not.toHaveAttribute('data-voice-speaking')
    expect(listen()).toBeEnabled()
  })

  it('falls back to the browser’s speech for the rest of the brief on a refusal code (tts_off)', async () => {
    const { root } = draw()
    await press()
    setup.server.refuse(0, 'tts_off', 503)
    await settle()

    expect(setup.browser.spoken()).toEqual(SENTENCES)
    expect(setup.session.serverless()).toBe(true)
    expect(setup.audio.plays).toHaveLength(0)
    expect(root).toHaveAttribute('data-voice-speaking', 'true')
    act(() => SENTENCES.forEach((_, index) => setup.browser.play(index)))
    expect(root).not.toHaveAttribute('data-voice-speaking')
  })

  it('says each sentence once, in order, with a row number joined to its title', () => {
    expect(spokenSentences(SPOKEN)).toEqual(SENTENCES)
    expect(spokenSentences('Morning. 10. Invoices. 11. Payroll.')).toEqual(['Morning.', '10. Invoices.', '11. Payroll.'])
    expect(spokenSentences('Morning, Sam')).toEqual(['Morning, Sam'])
    expect(spokenSentences('  ')).toEqual([])
  })

  it('logs and stays quiet when this browser cannot make a voice', () => {
    speech.make = vi.fn(() => {
      throw new Error('no Audio here')
    })
    const { root } = draw()
    fireEvent.click(listen())
    expect(console.warn).toHaveBeenCalled()
    expect(root).not.toHaveAttribute('data-voice-speaking')
    expect(listen()).toBeEnabled()
  })
})

describe('BriefVoice: Stop', () => {
  it('stops the audio and every request, and the late events do nothing', async () => {
    const { root } = draw()
    await press()
    setup.server.answer(0)
    await settle()
    act(() => setup.audio.plays[0].events.onPlaying())
    const stopsBefore = setup.audio.player.stop.mock.calls.length

    fireEvent.click(screen.getByRole('button', { name: STOP }))
    expect(setup.audio.player.stop.mock.calls.length).toBe(stopsBefore + 1)
    expect(setup.browser.port.cancel).toHaveBeenCalled()
    expect(setup.server.calls[1].signal.aborted).toBe(true)
    expect(root).not.toHaveAttribute('data-voice-speaking')
    expect(listen()).toBeEnabled()

    act(() => setup.audio.plays[0].events.onEnded())
    setup.server.answer(1)
    await settle()
    expect(setup.audio.plays).toHaveLength(1)
    expect(setup.server.texts()).toEqual(SENTENCES.slice(0, 2))
  })

  it('stops when it is taken off the page', async () => {
    const { view } = draw()
    await press()
    const stopsBefore = setup.audio.player.stop.mock.calls.length
    view.unmount()
    expect(setup.audio.player.stop.mock.calls.length).toBe(stopsBefore + 1)
    expect(setup.server.calls[0].signal.aborted).toBe(true)
  })

  it('reads the brief again from its start on the next press', async () => {
    draw()
    await press()
    fireEvent.click(screen.getByRole('button', { name: STOP }))
    await press()
    expect(setup.server.texts()).toEqual([...SENTENCES.slice(0, 2), ...SENTENCES.slice(0, 2)])
    expect(speech.make).toHaveBeenCalledTimes(1)
  })
})

describe("BriefVoice and the tab's voice session", () => {
  it('holds Listen while a hold or a call runs', () => {
    voice.current = hostVoiceStub(QUICK_LISTENING)
    draw()
    expect(listen()).toBeDisabled()
  })

  it('offers Listen again once the session is idle', () => {
    voice.current = hostVoiceStub({ snapshot: { mode: null, phase: 'idle', micOpen: false } })
    draw()
    expect(listen()).toBeEnabled()
  })

  it('stops a reading when the session starts, so Auto never talks over itself', async () => {
    const { root, view } = draw()
    await press()
    expect(root).toHaveAttribute('data-voice-speaking', 'true')
    const stopsBefore = setup.audio.player.stop.mock.calls.length
    voice.current = hostVoiceStub(QUICK_LISTENING)
    view.rerender(<BriefVoice spoken={SPOKEN} />)
    expect(setup.audio.player.stop.mock.calls.length).toBe(stopsBefore + 1)
    expect(setup.server.calls[0].signal.aborted).toBe(true)
    expect(root).not.toHaveAttribute('data-voice-speaking')
  })
})

describe('BriefVoice: the protected surface', () => {
  it('draws Listen inside the guarded brief-voice surface', () => {
    draw()
    const surface = document.querySelector<HTMLElement>(`[${PROTECTED_SURFACE_ATTR}="${BRIEF_VOICE}"]`)
    expect(surface).not.toBeNull()
    expect(surface).toContainElement(listen())
    const inner = surface?.querySelector<HTMLElement>(`[${PROTECTED_SURFACE_INNER_ATTR}]`)
    for (const element of [surface, inner]) {
      expect(element?.style.getPropertyValue('display')).toBe('block')
      expect(element?.style.getPropertyPriority('display')).toBe(IMPORTANT)
      expect(element?.style.getPropertyPriority('visibility')).toBe(IMPORTANT)
    }
  })

  it('puts the AI voice caption and Stop inside it while Auto reads, and the speaking flag outside it', async () => {
    const { root } = draw()
    await press()
    const surface = document.querySelector<HTMLElement>(`[${PROTECTED_SURFACE_ATTR}="${BRIEF_VOICE}"]`)
    expect(surface).toContainElement(screen.getByRole('note', { name: AI_VOICE_LABEL }))
    expect(surface).toContainElement(screen.getByRole('button', { name: STOP }))
    expect(root).toHaveAttribute('data-voice-speaking', 'true')
    expect(surface?.contains(root)).toBe(false)
    expect(screen.queryByRole('button', { name: LISTEN_TO_THE_BRIEF })).toBeNull()
  })

  it('makes every control a plain button', async () => {
    draw()
    expect(listen()).toHaveAttribute('type', 'button')
    await press()
    expect(screen.getByRole('button', { name: STOP })).toHaveAttribute('type', 'button')
  })
})
