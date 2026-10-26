// Theme voice — Studio on the host voice provider: the Live chip (components/chatbot/chat-mode-bar.tsx) is pressed
// while the tab's session is a live call, starts Live on a press and ends it on the next, says why Live can't start,
// and is never fenced on the AI label; the room's wave (components/voice/host-presence-orb.tsx) shows while a session
// runs, in its phase, reading the engine's levels each frame.
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import React from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

const voice = vi.hoisted(() => ({ current: null as unknown }))
const orb = vi.hoisted(() => ({ props: null as null | { state: string; levelsRef: { current: unknown }; horizon?: number } }))
vi.mock('@/components/voice/voice-provider', () => ({ useHostVoice: () => voice.current }))
vi.mock('@/components/voice/PresenceOrb', () => ({
  PresenceOrb: (props: { state: string; levelsRef: { current: unknown }; horizon?: number }) => {
    orb.props = props
    return <div data-testid="orb" data-orb-state={props.state} />
  },
}))

import { ChatModeBar } from '@/components/chatbot/chat-mode-bar'
import { HostPresenceOrb } from '@/components/voice/host-presence-orb'
import { type HostVoiceStubPatch, hostVoiceStub } from '@/components/voice/__tests__/support/host-voice-stub'
import { PAGE_AI_LABEL, useAiLabelStore } from '@/lib/ai-label/store'
import { LIVE_REASON_SENTENCES } from '@/lib/voice/live/refusal'

const NO_OP = () => undefined

function modeBar(patch: HostVoiceStubPatch) {
  const stub = hostVoiceStub(patch)
  voice.current = stub
  render(
    <ChatModeBar
      isCodeActive={false}
      isPlanActive={false}
      isMissionActive={false}
      onCodeClick={NO_OP}
      onPlanClick={NO_OP}
      onMissionClick={NO_OP}
    />,
  )
  return { stub, chip: screen.getByRole('button', { name: 'Live' }) }
}

afterEach(() => {
  cleanup()
  orb.props = null
  useAiLabelStore.getState().releaseLabel(PAGE_AI_LABEL)
})

describe("Studio's Live chip", () => {
  it('starts Auto Live on a press while no call runs', () => {
    const { stub, chip } = modeBar({ snapshot: { mode: null, phase: 'idle' } })
    expect(chip.getAttribute('aria-pressed')).toBe('false')
    fireEvent.click(chip)
    expect(stub.start).toHaveBeenCalledWith({ mode: 'live' })
    expect(stub.stop).not.toHaveBeenCalled()
  })

  it('is pressed during a live call, and a press ends it', () => {
    const { stub, chip } = modeBar({ snapshot: { mode: 'live', phase: 'speaking' } })
    expect(chip.getAttribute('aria-pressed')).toBe('true')
    fireEvent.click(chip)
    expect(stub.stop).toHaveBeenCalled()
    expect(stub.start).not.toHaveBeenCalled()
  })

  it("says why Live can't start, and still tries (the mint is the truth)", () => {
    const reason = LIVE_REASON_SENTENCES.voice_live_plan_required
    const { stub, chip } = modeBar({ snapshot: { mode: null, phase: 'idle' }, available: { live: false, reason } })
    expect(chip.getAttribute('title')).toBe(reason)
    expect(chip.getAttribute('data-live-available')).toBe('false')
    fireEvent.click(chip)
    expect(stub.start).toHaveBeenCalledWith({ mode: 'live' })
  })

  it('is never fenced on the AI label: hidden label, the chip still works', () => {
    useAiLabelStore.getState().setHidden(PAGE_AI_LABEL, true)
    const { stub, chip } = modeBar({ snapshot: { mode: null, phase: 'idle' } })
    expect(chip).not.toBeDisabled()
    expect(chip.closest('fieldset')).toBeNull()
    fireEvent.click(chip)
    expect(stub.start).toHaveBeenCalled()
  })
})

describe("Studio's presence wave", () => {
  it('shows while a session runs, in its phase, reading the levels each frame', () => {
    const stub = hostVoiceStub({ snapshot: { phase: 'speaking' }, levels: { user: 0.1, agent: 0.6 } })
    voice.current = stub
    render(<HostPresenceOrb horizon={0.74} />)
    expect(screen.getByTestId('orb').getAttribute('data-orb-state')).toBe('speaking')
    expect(orb.props?.horizon).toBe(0.74)
    expect(orb.props?.levelsRef.current).toEqual({ user: 0.1, agent: 0.6 })
    expect(orb.props?.levelsRef.current).toEqual({ user: 0.1, agent: 0.6 })
    expect(stub.readLevels).toHaveBeenCalledTimes(2)
  })

  it('draws nothing while no session runs', () => {
    voice.current = hostVoiceStub({ snapshot: { mode: null, phase: 'idle' } })
    render(<HostPresenceOrb />)
    expect(screen.queryByTestId('orb')).toBeNull()
  })
})
