/**
 * PRD-261 W-A6b — the boot route's entry data for the Design Kit's Play package and for the hello
 * test package (orchestrator serve.py theme_entry over each theme.json), for the tests of the
 * installed theme's registry entry.
 */
import type { InstalledThemeEntryData } from '@/lib/theme-host/boot-types'
import type { UiStyle } from '@/lib/theme-host/registry-types'

export const INSTALLED_PLAY_STYLE: UiStyle = 'theme:install:0b9a3c3e-6a1d-4f9e-9a43-0c1f2d3e4f50'
export const INSTALLED_HELLO_STYLE: UiStyle = 'theme:install:5d0f2a7e-1c4b-4e9a-b3d6-8f1a2c3e4b5d'

/** The kit's themes/play/theme.json, as the boot route answers it for a private upload. */
export const PLAY_PACKAGE_ENTRY: InstalledThemeEntryData = {
  name: 'Play',
  base: 'studio',
  htmlClass: 'play',
  minWidth: 1024,
  tone: 'from-look',
  reviewed: false,
  ownsPages: ['/chat', '/deliverables'],
  routes: ['conversations'],
  nav: [
    { id: 'nav-play-conversations', label: 'Conversations', path: '/x/conversations' },
    { id: 'nav-play-looks', label: 'Make it yours', path: '/looks' },
    { id: 'nav-play-style-sheet', label: 'Style sheet', path: '/looks/style' },
  ],
  slots: ['frame@1', 'lookEditor@1', 'styleSheet@1', 'notFound@1', 'asleep@1'],
}

/** orchestrator/tests/fixtures/themes/hello/theme.json: looks declared, a class of its own, the user's tone. */
export const HELLO_PACKAGE_ENTRY: InstalledThemeEntryData = {
  name: 'Hello theme',
  base: 'studio',
  htmlClass: 'hello',
  minWidth: 0,
  tone: 'user',
  reviewed: false,
  ownsPages: [],
  routes: ['hello'],
  nav: [{ id: 'hello', label: 'Hello', path: '/x/hello' }],
  slots: ['frame@1', 'lookEditor@1', 'styleSheet@1'],
}
