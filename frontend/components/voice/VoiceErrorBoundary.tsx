'use client'

/**
 * VoiceErrorBoundary — the voice layer may never take the page down again.
 *
 * Born from the two-session splice crash (a ReferenceError in the wave
 * white-screened the whole app the moment Live mounted). Any exception in
 * a voice surface is contained here: logged with a grep-able tag, rendered
 * as nothing (the chat beneath keeps working), reset when the user toggles
 * Live again.
 */

import { Component, type ReactNode } from 'react'

interface Props {
  children: ReactNode
}

interface State {
  crashed: boolean
}

export class VoiceErrorBoundary extends Component<Props, State> {
  state: State = { crashed: false }

  static getDerivedStateFromError(): State {
    return { crashed: true }
  }

  componentDidCatch(error: unknown, info: unknown) {
    // Grep-able in the console AND in any client-log forwarding.
    console.error('[VoiceUI] crashed — contained by VoiceErrorBoundary', error, info)
  }

  render() {
    if (this.state.crashed) return null
    return this.props.children
  }
}
