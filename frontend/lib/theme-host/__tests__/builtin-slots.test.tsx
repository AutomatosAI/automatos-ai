/**
 * PRD-261 A-04b — builtin Play's registrations (lib/theme-host/builtin/play-slots.tsx), read by
 * ref through lib/theme-host/builtin/slots.ts. They pass the loader's own registration checks, as
 * an installed theme's register(host) must (known slot ids, Q3 page routes, the route pattern, a
 * component each, a frame); they match Play's registry entry; and Play's frame@1 hands PlayFrame
 * the page and MainLayout's two frame@1 props.
 */
import React from 'react'
import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

vi.mock('@/components/play/shell/play-frame-lazy', () => ({
  PlayFrame: ({ children, fullBleed, autoWidget }: { children?: React.ReactNode; fullBleed?: boolean; autoWidget?: React.ReactNode }) => (
    <section data-testid="play-frame" data-full-bleed={String(fullBleed)}>
      {children}
      {autoWidget}
    </section>
  ),
}))

import { BUILTIN_PLAY_ENTRY } from '../builtin/play-entry'
import { BUILTIN_PLAY_REGISTRATIONS, PLAY_CONVERSATIONS_ROUTE } from '../builtin/play-slots'
import { builtinRegistrationsOf } from '../builtin/slots'
import { FRAME_SLOT } from '../constants'
import { registerTheme } from '../loader'
import type { SlotId, ThemeEntry } from '../registry-types'
import { SDK_VERSION } from '../sdk'
import type { ThemeDefinition, ThemeRegistrations } from '../types'

/** Slots Play's frame draws itself, so they are not registered on their own (AsleepGate). */
const DRAWN_BY_THE_FRAME: readonly SlotId[] = ['asleep@1']

/** A register(host) that hands the host every registration, as an installed theme's would. */
function definitionOf(registrations: ThemeRegistrations): ThemeDefinition {
  return {
    register(host) {
      registrations.slots.forEach((component, slot) => host.registerSlot(slot, component))
      registrations.pages.forEach((component, route) => host.registerPage(route, component))
      registrations.routes.forEach((component, path) => host.registerRoute(path, component))
    },
  }
}

describe("builtin Play's registrations", () => {
  it("pass the loader's registration checks, with Play's frame as frame@1", () => {
    const registered = registerTheme(definitionOf(BUILTIN_PLAY_REGISTRATIONS), SDK_VERSION)
    expect(registered.frame).toBe(BUILTIN_PLAY_REGISTRATIONS.frame)
    expect(registered.slots.get(FRAME_SLOT)).toBe(BUILTIN_PLAY_REGISTRATIONS.frame)
    expect(Array.from(registered.slots.keys()).sort()).toEqual(Array.from(BUILTIN_PLAY_REGISTRATIONS.slots.keys()).sort())
    expect(Array.from(registered.pages.keys())).toEqual(Array.from(BUILTIN_PLAY_REGISTRATIONS.pages.keys()))
    expect(Array.from(registered.routes.keys())).toEqual([PLAY_CONVERSATIONS_ROUTE])
  })

  it("match Play's registry entry: its pages, its routes and every slot it lists", () => {
    expect(Array.from(BUILTIN_PLAY_REGISTRATIONS.pages.keys())).toEqual(BUILTIN_PLAY_ENTRY.ownsPages)
    expect(Array.from(BUILTIN_PLAY_REGISTRATIONS.routes.keys())).toEqual(BUILTIN_PLAY_ENTRY.routes)
    const listed = BUILTIN_PLAY_ENTRY.slots.filter((slot) => !DRAWN_BY_THE_FRAME.includes(slot))
    expect(Array.from(BUILTIN_PLAY_REGISTRATIONS.slots.keys()).sort()).toEqual([...listed].sort())
  })

  it("register the frame, the look editor and the style sheet every theme ships (PA-14), and the 404", () => {
    for (const slot of [FRAME_SLOT, 'lookEditor@1', 'styleSheet@1', 'notFound@1'] as const) {
      expect(BUILTIN_PLAY_REGISTRATIONS.slots.has(slot), slot).toBe(true)
    }
  })

  it("hand Play's frame the page and MainLayout's full-bleed flag and Auto's widget (frame@1 props)", () => {
    const Frame = BUILTIN_PLAY_REGISTRATIONS.frame
    render(
      <Frame fullBleed autoWidget={<aside data-testid="auto-widget" />}>
        <div data-testid="page" />
      </Frame>,
    )
    const frame = screen.getByTestId('play-frame')
    expect(frame).toHaveAttribute('data-full-bleed', 'true')
    expect(frame).toContainElement(screen.getByTestId('page'))
    expect(frame).toContainElement(screen.getByTestId('auto-widget'))
  })
})

describe('builtinRegistrationsOf', () => {
  it("is builtin Play's registrations for Play's entry", () => {
    expect(builtinRegistrationsOf(BUILTIN_PLAY_ENTRY)).toBe(BUILTIN_PLAY_REGISTRATIONS)
  })

  it('is null for Classic and Studio (no entry) and for an installed theme', () => {
    const installed: ThemeEntry = { ...BUILTIN_PLAY_ENTRY, ref: 'install:00000000-0000-4000-8000-0000000a04b2', kind: 'installed' }
    expect(builtinRegistrationsOf(null)).toBeNull()
    expect(builtinRegistrationsOf(installed)).toBeNull()
    expect(builtinRegistrationsOf({ ...BUILTIN_PLAY_ENTRY, kind: 'installed' })).toBeNull()
  })
})
