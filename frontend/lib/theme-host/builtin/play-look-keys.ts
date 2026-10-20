/**
 * PRD-261 A-09b (PA-7) — where builtin Play keeps its looks in this browser: the PRD-260 keys,
 * `automatos-play.*`, unchanged. lib/looks/storage.ts is theme-neutral and takes its keys from the
 * active theme; an installed theme's are `automatos-theme.<themeId>.*` (installedLookKeys there),
 * builtin Play's are these, so no look a Play user already has moves before A-18 (whose shim reads
 * them once into the installed Play package's keys; A-19 deletes this file).
 *
 * On the server Play's looks are kept under its package id, `automatos/play`
 * (orchestrator/services/themes/looks.py BUILTIN_PLAY_THEME_ID), so A-18 finds them in place.
 */
import type { LookKeyBases, LookTheme } from '@/lib/looks/storage'

/** The theme id Play's looks are kept under on the server (GET/PUT /api/themes/looks/{themeId}). */
export const BUILTIN_PLAY_THEME_ID = 'automatos/play'

/**
 * Play's storage key bases (lib/looks/storage.ts adds the scope, F5):
 * - saved: the saved looks, a JSON array of Look objects;
 * - active: the active look, its id plus its values, so an unsaved edit survives a reload;
 * - pointer: the scope whose active look the boot script paints (it runs before anyone signs in);
 * - moved: the scope the unscoped, pre-F5 looks moved to (they move once, to the first account).
 */
export const BUILTIN_PLAY_LOOK_KEYS: LookKeyBases = Object.freeze({
  saved: 'automatos-play.looks',
  active: 'automatos-play.look',
  pointer: 'automatos-play.look.current',
  moved: 'automatos-play.looks.moved-to',
})

/** Builtin Play as the look store sees it: its server id and its legacy keys. */
export const BUILTIN_PLAY_LOOK_THEME: LookTheme = Object.freeze({
  themeId: BUILTIN_PLAY_THEME_ID,
  keys: BUILTIN_PLAY_LOOK_KEYS,
})
