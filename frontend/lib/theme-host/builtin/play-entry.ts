/**
 * PRD-261 A-02a — builtin Play (PRD-260) as one entry of the theme registry.
 *
 * Play is compiled into the app until it ships as a package (A-17), so its entry lives here.
 * This folder is the one place in the core, outside Play's own folders, that names Play (PA-7).
 * PA-8: Play keeps `play` as its class on <html>, so its stylesheet (app/play-theme.css,
 * `html.play …`) is unchanged.
 */
import { LOOK_STYLE_HREF, LOOKS_HREF } from '@/lib/ticket-links'

import type { ThemeEntry, UiStyle } from '../registry-types'

export const BUILTIN_PLAY_REF = 'builtin:play'
export const BUILTIN_PLAY_STYLE = 'theme:builtin:play'
export const BUILTIN_PLAY_HTML_CLASS = 'play'

/**
 * PRD-261 W-A6b: whether the one look runtime the host has (Play's looks) runs under a theme: the
 * theme whose <html> class is Play's, builtin Play or Play's package. Only such a theme keeps the
 * look's tone cookie, so only its boot may start on it (boot.ts); the LookProvider paints only under it.
 */
export function runsPlayLooks(theme: { readonly htmlClass: string } | null | undefined): boolean {
  return theme?.htmlClass === BUILTIN_PLAY_HTML_CLASS
}

/** Play's Conversations page: its theme route `conversations` (PRD-261 A-04b); /conversations redirects here. */
export const PLAY_CONVERSATIONS_ROUTE_NAME = 'conversations'
export const PLAY_CONVERSATIONS_HREF = `/x/${PLAY_CONVERSATIONS_ROUTE_NAME}`

/** PRD-260 D7: Play's frame and pages render from this width up; below it, the core's fallback. */
export const BUILTIN_PLAY_MIN_WIDTH_PX = 1024

/**
 * Style cookies written before the axis took theme refs said `play`. They still mean builtin Play,
 * so a browser that chose Play keeps it. Read only through an own-property check (registry.ts).
 */
export const LEGACY_STYLE_ALIASES: Readonly<Record<string, UiStyle>> = { play: BUILTIN_PLAY_STYLE }

export const BUILTIN_PLAY_ENTRY: ThemeEntry = {
  ref: BUILTIN_PLAY_REF,
  name: 'Play',
  base: 'studio',
  htmlClass: BUILTIN_PLAY_HTML_CLASS,
  minWidth: BUILTIN_PLAY_MIN_WIDTH_PX,
  tone: 'from-look',
  kind: 'builtin',
  reviewed: true,
  ownsPages: ['/chat', '/deliverables'],
  routes: [PLAY_CONVERSATIONS_ROUTE_NAME],
  // PRD-260 M90: Play's own pages, listed in the search dialog only while Play is the style.
  nav: [
    { id: 'nav-play-conversations', label: 'Conversations', path: PLAY_CONVERSATIONS_HREF },
    { id: 'nav-play-looks', label: 'Make it yours', path: LOOKS_HREF },
    { id: 'nav-play-style-sheet', label: 'Style sheet', path: LOOK_STYLE_HREF },
  ],
  slots: ['frame@1', 'lookEditor@1', 'styleSheet@1', 'notFound@1', 'asleep@1'],
}
