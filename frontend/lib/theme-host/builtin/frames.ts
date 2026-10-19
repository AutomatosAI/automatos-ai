/**
 * PRD-261 A-02b — each builtin theme's frame, by the theme's ref. MainLayout reads it by the active
 * theme's ref and never names a builtin theme; every other theme (an installed one) draws through
 * the theme host's frame slot (components/theme-host/theme-frame-slot.tsx). Each frame is a
 * next/dynamic slot, so its chunk is fetched only when it renders. A-04b moves builtin Play's frame
 * into the frame@1 slot, and this map goes.
 */
import type { ComponentType, ReactNode } from 'react'

import { PlayFrame } from '@/components/play/shell/play-frame-lazy'

import type { ThemeEntry } from '../registry-types'
import type { FrameProps } from '../types'
import { BUILTIN_PLAY_REF } from './play-entry'

/** frame@1's props, with the page always passed (MainLayout always has one). */
export type BuiltinFrame = ComponentType<FrameProps & { readonly children: ReactNode }>

export const BUILTIN_THEME_FRAMES: ReadonlyMap<string, BuiltinFrame> = new Map<string, BuiltinFrame>([
  [BUILTIN_PLAY_REF, PlayFrame],
])

/** The frame of the active theme when it is a builtin one; null for Classic, Studio and an installed theme. */
export function builtinFrameOf(entry: ThemeEntry | null): BuiltinFrame | null {
  return entry === null ? null : BUILTIN_THEME_FRAMES.get(entry.ref) ?? null
}
