// Theme voice — the voice button core draws for themes (components/voice/voice-button.tsx) over a HostVoice stub: the
// press as a hold and as a toggle, the hotkey (never while the member types), the microphone prompt, a failure's
// sentence with Try again, the "AI voice assistant" caption and the stop control while the session is the button's,
// inside the guarded `voice-control` surface, the data attributes and custom properties a theme styles it by, the
// state for screen readers, and a button that says voice isn't here outside a provider. Never fenced on the AI label,
// and inside a form (Studio's composer) none of its controls submits it.
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import React from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

const voice = vi.hoisted(() => ({ current: null as unknown }))
vi.mock('@/components/voice/voice-provider', () => ({ useOptionalHostVoice: () => voice.current }))

import { type HostVoiceStubPatch, hostVoiceStub } from '@/components/voice/__tests__/support/host-voice-stub'
import { AI_VOICE_LABEL } from '@/components/voice/AiVoiceLabel'
import { PROTECTED_SURFACE_ATTR, PROTECTED_SURFACE_INNER_ATTR } from '@/components/theme-host/protected-surface'
import { VoiceButton, type VoiceButtonProps } from '@/components/voice/voice-button'
import { VOICE_BUTTON_CSS_VARS, VOICE_BUTTON_WORDS } from '@/components/voice/voice-button-view'
import { CLOSE, CLOSE_LIVE_VOICE, END_CALL, MIC_LIVE } from '@/components/voice/voice-controls'
import { PAGE_AI_LABEL, useAiLabelStore } from '@/lib/ai-label/store'
import { QUICK_ASK_ERRORS } from '@/lib/voice/quick-ask/copy'

const IDLE: HostVoiceStubPatch = { snapshot: { mode: null, phase: 'idle', micOpen: false, chatId: null } }
const QUICK_LISTENING: HostVoiceStubPatch = { snapshot: { mode: 'quick', phase: 'listening', micOpen: true } }
const VOICE_CONTROL = 'voice-control'
const IMPORTANT = 'important'

function draw(patch: HostVoiceStubPatch | null, props: VoiceButtonProps = {}) {
  const stub = patch === null ? null : hostVoiceStub(patch)
  voice.current = stub
  const view = render(<VoiceButton {...props} />)
  const press = view.container.querySelector<HTMLButtonElement>('[data-voice-button="press"]')
  const root = view.container.querySelector<HTMLElement>('[data-voice-button="root"]')
  if (!press || !root) throw new Error('no voice button drawn')
  return { stub, press, root, view }
}

function controlSurface(): HTMLElement | null {
  return document.querySelector<HTMLElement>(`[${PROTECTED_SURFACE_ATTR}="${VOICE_CONTROL}"]`)
}

/** A labelled control: a plain button showing its word beside its icon, never the bare icon. */
function expectLabelled(control: HTMLElement, word: string): void {
  expect(control).toHaveAttribute('type', 'button')
  expect(control.textContent).toBe(word)
  expect(control.querySelector('svg')).not.toBeNull()
}

afterEach(() => {
  cleanup()
  voice.current = null
  useAiLabelStore.getState().releaseLabel(PAGE_AI_LABEL)
})

describe('the press', () => {
  it('holds by default: the pointer going down starts (auto), letting go sends', () => {
    const { stub, press } = draw(IDLE)
    expect(press.getAttribute('aria-label')).toBe(VOICE_BUTTON_WORDS.hold)
    fireEvent.pointerDown(press)
    expect(stub?.start).toHaveBeenCalledWith({ mode: 'auto' })
    fireEvent.pointerUp(press)
    expect(stub?.release).toHaveBeenCalledTimes(1)
  })

  it('a toggle starts on a click and sends on the next while the microphone is open', () => {
    const idle = draw(IDLE, { press: 'toggle', mode: 'quick', conversation: 'chat-1' })
    expect(idle.press.getAttribute('aria-label')).toBe(VOICE_BUTTON_WORDS.toggle)
    fireEvent.click(idle.press)
    expect(idle.stub?.start).toHaveBeenCalledWith({ mode: 'quick', chatId: 'chat-1' })
    cleanup()
    const hearing = draw(QUICK_LISTENING, { press: 'toggle', mode: 'quick', conversation: 'chat-1' })
    fireEvent.click(hearing.press)
    expect(hearing.stub?.release).toHaveBeenCalledTimes(1)
  })

  it('takes the theme’s label as its name', () => {
    const { press } = draw(IDLE, { label: 'Ask Auto' })
    expect(press.getAttribute('aria-label')).toBe('Ask Auto')
  })

  it('is never fenced on the AI label: with the label hidden, the press still starts', () => {
    useAiLabelStore.getState().setHidden(PAGE_AI_LABEL, true)
    const { stub, press } = draw(IDLE, { press: 'toggle' })
    expect(press).not.toBeDisabled()
    expect(press.closest('fieldset')).toBeNull()
    fireEvent.click(press)
    expect(stub?.start).toHaveBeenCalled()
  })
})

describe('the hotkey', () => {
  it('holds on Space anywhere but a field: down starts, up sends', () => {
    const { stub } = draw(IDLE)
    fireEvent.keyDown(document.body, { code: 'Space' })
    expect(stub?.start).toHaveBeenCalledWith({ mode: 'auto' })
    fireEvent.keyUp(document.body, { code: 'Space' })
    expect(stub?.release).toHaveBeenCalledTimes(1)
  })

  it('leaves Space to the composer, and to a control that has focus', () => {
    const { stub } = draw(IDLE)
    const composer = document.createElement('textarea')
    const other = document.createElement('button')
    document.body.append(composer, other)
    try {
      fireEvent.keyDown(composer, { code: 'Space' })
      fireEvent.keyDown(other, { code: 'Space' })
      expect(stub?.start).not.toHaveBeenCalled()
    } finally {
      composer.remove()
      other.remove()
    }
  })

  it('ignores a held key’s repeats, other keys, and a key with a modifier', () => {
    const { stub } = draw(IDLE)
    fireEvent.keyDown(document.body, { code: 'Space', repeat: true })
    fireEvent.keyDown(document.body, { code: 'KeyV' })
    fireEvent.keyDown(document.body, { code: 'Space', ctrlKey: true })
    expect(stub?.start).not.toHaveBeenCalled()
  })

  it('takes the theme’s key, and none at all for null', () => {
    const custom = draw(IDLE, { hotkey: 'KeyV' })
    fireEvent.keyDown(document.body, { code: 'Space' })
    fireEvent.keyDown(document.body, { code: 'KeyV' })
    expect(custom.stub?.start).toHaveBeenCalledTimes(1)
    cleanup()
    const none = draw(IDLE, { hotkey: null })
    fireEvent.keyDown(document.body, { code: 'Space' })
    expect(none.stub?.start).not.toHaveBeenCalled()
  })
})

describe('while the session is the button’s', () => {
  it('shows the AI voice caption and Stop inside the guarded voice-control surface', () => {
    const { stub } = draw(QUICK_LISTENING)
    const surface = controlSurface()
    expect(surface).not.toBeNull()
    const caption = within(surface as HTMLElement).getByTestId('ai-voice-label')
    expect(caption).toHaveTextContent(AI_VOICE_LABEL)
    expect(caption.getAttribute('role')).toBe('note')
    const stop = within(surface as HTMLElement).getByRole('button', { name: VOICE_BUTTON_WORDS.stop })
    expectLabelled(stop, VOICE_BUTTON_WORDS.stop)
    fireEvent.click(stop)
    expect(stub?.stop).toHaveBeenCalledTimes(1)
  })

  it('lets the caption, Mic live and Stop wrap, so the row fits a phone under the button', () => {
    draw({ snapshot: { mode: 'quick', phase: 'listening', micOpen: true, durationSec: 12 } })
    const surface = controlSurface() as HTMLElement
    const row = within(surface).getByTestId('ai-voice-label').parentElement as HTMLElement
    expect(row.className).toContain('flex-wrap')
    expect(row).toHaveTextContent(`${AI_VOICE_LABEL}${MIC_LIVE}0:12${VOICE_BUTTON_WORDS.stop}`)
  })

  it('guards the surface and its inner wrapper inline with !important', () => {
    draw(QUICK_LISTENING)
    const host = controlSurface() as HTMLElement
    const inner = host.querySelector<HTMLElement>(`[${PROTECTED_SURFACE_INNER_ATTR}]`) as HTMLElement
    for (const element of [host, inner]) {
      for (const property of ['display', 'visibility', 'opacity', 'pointer-events', 'position']) {
        expect({ property, priority: element.style.getPropertyPriority(property) }).toEqual({ property, priority: IMPORTANT })
      }
      expect(element.style.getPropertyValue('visibility')).toBe('visible')
      expect(element.style.getPropertyValue('opacity')).toBe('1')
    }
  })

  it('shows no caption and no control while no session is the button’s', () => {
    draw(IDLE)
    expect(controlSurface()).toBeNull()
    expect(screen.queryByTestId('ai-voice-label')).toBeNull()
  })

  it('leaves a session on another conversation alone: the button stays idle', () => {
    const { root } = draw({ snapshot: { mode: 'quick', phase: 'listening', micOpen: true, chatId: 'chat-2' } }, { conversation: 'chat-1' })
    expect(root.getAttribute('data-voice-state')).toBe('idle')
    expect(controlSurface()).toBeNull()
  })

  it('a live call has mute and End call in the surface, and the provider’s silent-microphone notice', () => {
    const { stub } = draw({ snapshot: { mode: 'live', phase: 'listening', micOpen: true }, live: { micSilent: true } })
    const surface = controlSurface() as HTMLElement
    fireEvent.click(within(surface).getByRole('button', { name: 'Mute microphone' }))
    const end = within(surface).getByRole('button', { name: END_CALL })
    expectLabelled(end, END_CALL)
    fireEvent.click(end)
    expect(stub?.live.toggleMute).toHaveBeenCalledTimes(1)
    expect(stub?.stop).toHaveBeenCalledTimes(1)
    expect(screen.getByTestId('mic-silent-banner')).toBeInTheDocument()
  })

  it('an ended call has Close in the surface, its word on screen', () => {
    const { stub } = draw({ snapshot: { mode: 'live', phase: 'ended', micOpen: false } })
    const close = within(controlSurface() as HTMLElement).getByRole('button', { name: CLOSE_LIVE_VOICE })
    expectLabelled(close, CLOSE)
    fireEvent.click(close)
    expect(stub?.stop).toHaveBeenCalledTimes(1)
  })
})

describe('the prompt, a failure and the screen reader', () => {
  it('names the browser’s microphone prompt while the session waits for it', () => {
    draw({ snapshot: { mode: 'quick', phase: 'connecting', micOpen: false } })
    expect(screen.getByText(VOICE_BUTTON_WORDS.allowMic)).toBeInTheDocument()
  })

  it('shows a failure’s sentence with Try again (the same engine) and Close', () => {
    const { stub } = draw({ snapshot: { mode: 'quick', phase: 'error', micOpen: false, error: QUICK_ASK_ERRORS.permission } })
    expect(screen.getAllByText(QUICK_ASK_ERRORS.permission).length).toBeGreaterThan(0)
    fireEvent.click(screen.getByRole('button', { name: 'Try again' }))
    expect(stub?.start).toHaveBeenCalledWith({ mode: 'quick' })
    fireEvent.click(screen.getByRole('button', { name: 'Close' }))
    expect(stub?.stop).toHaveBeenCalledTimes(1)
    expect(within(controlSurface() as HTMLElement).getByTestId('ai-voice-label')).toBeInTheDocument()
  })

  it('tells screen readers the state and that the microphone is live', () => {
    draw(QUICK_LISTENING)
    expect(screen.getByRole('status')).toHaveTextContent(`Auto is listening. ${MIC_LIVE}`)
  })
})

describe('what a theme styles it by', () => {
  it('carries the session’s state and engine, the press, the size and the microphone on the root', () => {
    const { root } = draw(QUICK_LISTENING, { press: 'toggle', size: 'lg' })
    expect({ ...root.dataset }).toMatchObject({
      voiceState: 'listening', voiceMode: 'quick', voicePress: 'toggle', voiceSize: 'lg', voiceMic: 'open',
    })
    cleanup()
    const idle = draw({ ...IDLE, available: { live: false, reason: null } })
    expect({ ...idle.root.dataset }).toMatchObject({ voiceState: 'idle', voiceMode: 'quick', voicePress: 'hold', voiceMic: 'closed' })
  })

  it('reads every --voice-button-* custom property, at rest and while active', () => {
    const idle = draw(IDLE).press.className
    cleanup()
    const active = draw(QUICK_LISTENING).press.className
    for (const name of VOICE_BUTTON_CSS_VARS) expect(`${idle} ${active}`).toContain(`var(${name},`)
  })
})

describe('inside a form (Studio’s composer, a theme’s own)', () => {
  const SESSIONS: Array<[string, HostVoiceStubPatch, string[]]> = [
    ['a quick ask', QUICK_LISTENING, [VOICE_BUTTON_WORDS.stop]],
    ['a failure', { snapshot: { mode: 'quick', phase: 'error', micOpen: false, error: QUICK_ASK_ERRORS.noSpeech } }, ['Try again', 'Close']],
    ['a live call', { snapshot: { mode: 'live', phase: 'listening', micOpen: true } }, ['Mute microphone', 'End call']],
    ['an ended call', { snapshot: { mode: 'live', phase: 'ended', micOpen: false } }, ['Close live voice']],
  ]

  it.each(SESSIONS)('submits nothing from %s’s controls: each is a plain button', (_name, patch, names) => {
    const submitted = vi.fn((event: React.FormEvent) => event.preventDefault())
    voice.current = hostVoiceStub(patch)
    render(<form onSubmit={submitted}><VoiceButton /></form>)
    for (const name of names) {
      const control = screen.getByRole('button', { name })
      expect(control).toHaveAttribute('type', 'button')
      fireEvent.click(control)
    }
    expect(submitted).not.toHaveBeenCalled()
  })
})

describe('outside a voice provider', () => {
  it('is drawn disabled and idle, saying voice isn’t available here, and never throws', () => {
    const { press, root } = draw(null)
    expect(press).toBeDisabled()
    expect(press.getAttribute('title')).toBe(VOICE_BUTTON_WORDS.unavailable)
    expect(root.getAttribute('data-voice-state')).toBe('idle')
    fireEvent.keyDown(document.body, { code: 'Space' })
    expect(controlSurface()).toBeNull()
  })
})
