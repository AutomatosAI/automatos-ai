// Theme voice — the one press every voice control shares (components/voice/voice-press.ts): what a press does in
// each state of the tab's session, a toggle and a hold over a HostVoice, the conversation a press talks into, and
// nothing at all without the host's voice.
import { cleanup, createEvent, fireEvent, render, screen } from '@testing-library/react'
import React from 'react'
import { afterEach, describe, expect, it } from 'vitest'

import { type HostVoiceStubPatch, hostVoiceStub } from '@/components/voice/__tests__/support/host-voice-stub'
import type { HostVoice } from '@/components/voice/voice-provider'
import {
  type VoicePressOptions,
  ownsSession,
  pressAction,
  useVoicePress,
} from '@/components/voice/voice-press'
import { IDLE_VOICE, type VoiceSnapshot } from '@/lib/voice/engine'

const TOGGLE_QUICK: VoicePressOptions = { mode: 'quick', press: 'toggle' }
const HOLD_QUICK: VoicePressOptions = { mode: 'quick', press: 'hold' }
const POINTER_CLICK = { detail: 1 }
const RIGHT_BUTTON = 2

function snapshot(patch: Partial<VoiceSnapshot>): VoiceSnapshot {
  return { ...IDLE_VOICE, ...patch }
}

function PressButton({ voice, options }: { voice: HostVoice | null; options: VoicePressOptions }) {
  const { active, handlers } = useVoicePress(voice, options)
  return (
    <button type="button" aria-pressed={active} {...handlers}>
      talk
    </button>
  )
}

function pressButton(patch: HostVoiceStubPatch, options: VoicePressOptions) {
  const stub = hostVoiceStub(patch)
  render(<PressButton voice={stub} options={options} />)
  return { stub, button: screen.getByRole('button', { name: 'talk' }) }
}

afterEach(() => {
  cleanup()
})

describe('what a press does', () => {
  it('starts a session when none is the control’s', () => {
    expect(pressAction(IDLE_VOICE, TOGGLE_QUICK)).toBe('start')
    expect(pressAction(snapshot({ mode: 'quick', phase: 'idle', heard: 'kept' }), TOGGLE_QUICK)).toBe('start')
  })

  it('ends a live call, whatever its phase', () => {
    const live: VoicePressOptions = { mode: 'live', press: 'toggle' }
    for (const phase of ['connecting', 'listening', 'speaking', 'ended', 'error'] as const) {
      expect(pressAction(snapshot({ mode: 'live', phase }), live)).toBe('stop')
    }
  })

  it('sends what was heard on a toggle while quick-ask hears, and does nothing more on a hold', () => {
    for (const phase of ['connecting', 'listening'] as const) {
      expect(pressAction(snapshot({ mode: 'quick', phase }), TOGGLE_QUICK)).toBe('release')
      expect(pressAction(snapshot({ mode: 'quick', phase }), HOLD_QUICK)).toBe('none')
    }
  })

  it('starts a new turn while Auto replies or after a failure (quick-ask cuts Auto off)', () => {
    for (const phase of ['thinking', 'speaking', 'error'] as const) {
      expect(pressAction(snapshot({ mode: 'quick', phase }), TOGGLE_QUICK)).toBe('start')
    }
  })

  it('owns only a session of its engine, on its conversation when it names one', () => {
    const quickOnChat2 = snapshot({ mode: 'quick', phase: 'listening', chatId: 'chat-2' })
    expect(ownsSession(snapshot({ mode: 'live', phase: 'listening' }), { mode: 'quick' })).toBe(false)
    expect(ownsSession(snapshot({ mode: 'live', phase: 'listening' }), { mode: 'auto' })).toBe(true)
    expect(ownsSession(quickOnChat2, { mode: 'quick', chatId: 'chat-1' })).toBe(false)
    expect(ownsSession(quickOnChat2, { mode: 'quick', chatId: 'chat-2' })).toBe(true)
    expect(ownsSession(quickOnChat2, { mode: 'quick', chatId: null })).toBe(true)
    expect(ownsSession(quickOnChat2, { mode: 'quick' })).toBe(true)
    expect(pressAction(quickOnChat2, { ...TOGGLE_QUICK, chatId: 'chat-1' })).toBe('start')
  })
})

describe('a toggle', () => {
  it('starts on a click, on the conversation it names', () => {
    const { stub, button } = pressButton({ snapshot: { mode: null, phase: 'idle' } }, { ...TOGGLE_QUICK, chatId: 'chat-4' })
    expect(button.getAttribute('aria-pressed')).toBe('false')
    fireEvent.click(button)
    expect(stub.start).toHaveBeenCalledWith({ mode: 'quick', chatId: 'chat-4' })
  })

  it('starts with the mode alone when it names no conversation', () => {
    const { stub, button } = pressButton({ snapshot: { mode: null, phase: 'idle' } }, { mode: 'auto', press: 'toggle' })
    fireEvent.click(button)
    expect(stub.start).toHaveBeenCalledWith({ mode: 'auto' })
  })

  it('sends on the next click while the microphone is open', () => {
    const { stub, button } = pressButton({ snapshot: { mode: 'quick', phase: 'listening' } }, TOGGLE_QUICK)
    expect(button.getAttribute('aria-pressed')).toBe('true')
    fireEvent.click(button)
    expect(stub.release).toHaveBeenCalledTimes(1)
    expect(stub.start).not.toHaveBeenCalled()
  })

  it('ignores the pointer: the click is the whole press', () => {
    const { stub, button } = pressButton({ snapshot: { mode: null, phase: 'idle' } }, TOGGLE_QUICK)
    fireEvent.pointerDown(button)
    fireEvent.pointerUp(button)
    expect(stub.start).not.toHaveBeenCalled()
    expect(stub.release).not.toHaveBeenCalled()
  })
})

describe('a hold', () => {
  it('starts on the pointer going down and sends once on letting go; the pointer’s click does nothing', () => {
    const { stub, button } = pressButton({ snapshot: { mode: null, phase: 'idle' } }, HOLD_QUICK)
    fireEvent.pointerDown(button)
    expect(stub.start).toHaveBeenCalledWith({ mode: 'quick' })
    fireEvent.pointerUp(button)
    fireEvent.pointerCancel(button)
    fireEvent.click(button, POINTER_CLICK)
    expect(stub.release).toHaveBeenCalledTimes(1)
    expect(stub.start).toHaveBeenCalledTimes(1)
  })

  it('ends when the window loses focus, as no pointer event says so; a blur with nothing held does nothing', () => {
    const { stub, button } = pressButton({ snapshot: { mode: null, phase: 'idle' } }, HOLD_QUICK)
    fireEvent.blur(window)
    expect(stub.release).not.toHaveBeenCalled()
    fireEvent.pointerDown(button)
    fireEvent.blur(window)
    expect(stub.release).toHaveBeenCalledTimes(1)
    fireEvent.pointerUp(button)
    expect(stub.release).toHaveBeenCalledTimes(1)
  })

  it('a right-button press does nothing', () => {
    const { stub, button } = pressButton({ snapshot: { mode: null, phase: 'idle' } }, HOLD_QUICK)
    // jsdom has no PointerEvent, so the event's button is set on it as a browser would.
    const press = createEvent.pointerDown(button)
    Object.defineProperty(press, 'button', { value: RIGHT_BUTTON })
    fireEvent(button, press)
    fireEvent.pointerUp(button)
    expect(stub.start).not.toHaveBeenCalled()
    expect(stub.release).not.toHaveBeenCalled()
  })

  it('a click no pointer made (a key, a screen reader) toggles, so nobody is shut out', () => {
    const idle = pressButton({ snapshot: { mode: null, phase: 'idle' } }, HOLD_QUICK)
    fireEvent.click(idle.button)
    expect(idle.stub.start).toHaveBeenCalledWith({ mode: 'quick' })
    cleanup()
    const hearing = pressButton({ snapshot: { mode: 'quick', phase: 'listening' } }, HOLD_QUICK)
    fireEvent.click(hearing.button)
    expect(hearing.stub.release).toHaveBeenCalledTimes(1)
  })

  it('a held Live press starts the call, and the next press ends it', () => {
    const live: VoicePressOptions = { mode: 'live', press: 'hold' }
    const idle = pressButton({ snapshot: { mode: null, phase: 'idle' } }, live)
    fireEvent.pointerDown(idle.button)
    expect(idle.stub.start).toHaveBeenCalledWith({ mode: 'live' })
    cleanup()
    const onTheCall = pressButton({ snapshot: { mode: 'live', phase: 'listening' } }, live)
    fireEvent.pointerDown(onTheCall.button)
    fireEvent.pointerUp(onTheCall.button)
    expect(onTheCall.stub.stop).toHaveBeenCalledTimes(1)
    expect(onTheCall.stub.release).not.toHaveBeenCalled()
  })
})

describe('without the host’s voice', () => {
  it('is never active and does nothing', () => {
    render(<PressButton voice={null} options={HOLD_QUICK} />)
    const button = screen.getByRole('button', { name: 'talk' })
    fireEvent.pointerDown(button)
    fireEvent.click(button)
    expect(button.getAttribute('aria-pressed')).toBe('false')
  })
})
