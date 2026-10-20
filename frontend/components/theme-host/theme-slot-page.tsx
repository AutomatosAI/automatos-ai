'use client'

import { Suspense, useEffect, useRef, type ReactNode } from 'react'

import { MainLayout } from '@/components/layout/main-layout'
import { refocusWhenShown } from '@/lib/theme-host/focus'
import { takeHeadingFocusAfterReload } from '@/lib/theme-host/switch-focus'
import type { SlotId } from '@/lib/theme-host/registry-types'

import { registeredComponent, useActiveThemeSlots } from './theme-slot-registrations'

/**
 * PRD-261 A-04b (PA-14) — a page only a theme draws, through one of its slots: Make it yours at
 * /looks (lookEditor@1) and the style sheet at /looks/style (styleSheet@1). Every theme ships both
 * (PA-14), so under a theme its screen renders full bleed inside MainLayout (the theme's frame).
 * Only Classic and Studio (no theme, or an installed theme that is off or failed, so Studio shows)
 * get the page's `note`; under a theme below its width, where its pages don't render (PRD-260 D7),
 * the `narrowNote`. While the width isn't known (the server, hydration) the screen goes to
 * MainLayout, which renders nothing until it is (P260-FX-25, M69); the screen is only its busy
 * heading then anyway (M72).
 *
 * After a note offered the switch to a theme, the screen's heading (`headingSelector`, an element
 * with tabIndex={-1}) takes the focus that dropped to <body> when the note went, as soon as the
 * screen renders, however late its code loads (P260-FX-22, M21). A switch onto an installed theme
 * reloads the page (PA-13), so the note's switch leaves word in this tab's sessionStorage just
 * before the reload, and the next load takes it once, as if the note had been shown (G3,
 * lib/theme-host/switch-focus.ts).
 */

export interface ThemeSlotPageProps {
  /** The slot whose screen this page is. */
  readonly slot: SlotId
  /** Under Classic and Studio, or a theme that registered no such slot: the core's note. */
  readonly note: ReactNode
  /** Under a theme, below its width: why its screen is not here. */
  readonly narrowNote: ReactNode
  /** The screen's heading, focused once it replaces a note that offered the switch. */
  readonly headingSelector: string
}

/** Remembers that a note was shown (on this load, or before the switch reloaded it); once the theme's screen shows, its heading takes focus. */
function useHeadingFocusAfterSwitch(themed: boolean, showsScreen: boolean, headingSelector: string): void {
  const offered = useRef(false)
  // G3: declared first, so the word left before the reload is taken before the screen is looked for.
  useEffect(() => {
    if (takeHeadingFocusAfterReload()) offered.current = true
  }, [])
  useEffect(() => {
    if (!themed) {
      offered.current = true
      return undefined
    }
    if (!showsScreen || !offered.current) return undefined
    offered.current = false
    return refocusWhenShown(headingSelector)
  }, [themed, showsScreen, headingSelector])
}

export function ThemeSlotPage({ slot, note, narrowNote, headingSelector }: ThemeSlotPageProps) {
  const { layout, registrations } = useActiveThemeSlots()
  const Screen = registeredComponent(registrations?.slots.get(slot))
  useHeadingFocusAfterSwitch(layout !== 'none', layout === 'theme', headingSelector)
  if (layout === 'none' || (registrations !== null && Screen === null)) return <MainLayout>{note}</MainLayout>
  if (layout === 'narrow') return <MainLayout>{narrowNote}</MainLayout>
  return (
    <MainLayout fullBleed>
      {Screen === null ? null : (
        <Suspense fallback={null}>
          <Screen />
        </Suspense>
      )}
    </MainLayout>
  )
}
