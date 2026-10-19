/**
 * PRD-261 A-03 — each builtin theme's client module (its stylesheet), by the theme's ref. Each is
 * its own chunk, fetched only while that theme is the style: rendered on the server, next/dynamic
 * writes the chunk's stylesheet link into the first response, so the theme's first paint is
 * styled, and a Studio or Classic page requests none of it.
 *
 * The map lives in this folder so the core (components/providers.tsx) reads it by ref and never
 * names a builtin theme (A-02b contract); adding or removing a builtin theme edits only this folder.
 */
import dynamic from 'next/dynamic'
import type { ComponentType } from 'react'

import { BUILTIN_PLAY_REF } from './play-entry'

export const BUILTIN_THEME_CLIENTS: ReadonlyMap<string, ComponentType> = new Map([
  [BUILTIN_PLAY_REF, dynamic(() => import('./play-client'))],
])
