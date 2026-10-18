/**
 * PRD-261 A-01, A-04a — the theme host's fixed values.
 *
 * An installed theme's description comes from the backend's active-theme route, and its files
 * from the same-origin files route under its package digest, so no theme URL ever comes from a
 * user: the manifest's file names must resolve under that folder (manifest.ts).
 */
import { STYLE_HTML_CLASS_NAMES } from '@/lib/ui-style'

import type { SlotId } from './registry-types'

/**
 * PA-2: the host contract's version (slots, pages, routes, the boot contract). A-06b moves it
 * into shared-modules.json next to sdkVersion, with the Python parity test.
 */
export const THEME_HOST_VERSION = '1.0.0'

/** The caller's active theme in this workspace (A-07b), read through the backend client. */
export const THEME_ACTIVE_ENDPOINT = '/api/themes/active'

/** An installed theme's files, by package digest, on the page's own origin (A-08a, proxied by A-08b). */
export const THEME_FILES_PATH = '/api/themes/files/'

/** A package digest: sha256 in lower-case hex (A-06a digest.py). */
export const THEME_DIGEST_PATTERN = /^[0-9a-f]{64}$/

/** How long each request, the entry module and the stylesheet may take before the load counts as failed. */
export const THEME_LOAD_TIMEOUT_MS = 10_000

/** `?safe_mode=themes` turns theme loading off for this tab (PRD-261 §6). */
export const SAFE_MODE_PARAM = 'safe_mode'
export const SAFE_MODE_VALUE = 'themes'
export const SAFE_MODE_STORAGE_KEY = 'automatos-theme-safe-mode'

/** A theme's own class on <html>: 3 to 32 characters, lower-case letters, digits and hyphens. */
export const HTML_CLASS_PATTERN = /^[a-z][a-z0-9-]{2,31}$/

/** Classes the core owns on <html> (every Style-axis class, and the tone's), never a theme's. */
export const RESERVED_HTML_CLASSES: readonly string[] = [...STYLE_HTML_CLASS_NAMES, 'classic', 'dark', 'light']

/** Mirrors of the host's status and failure reason on <html>, for the browser lane to wait on. */
export const THEME_HOST_STATUS_ATTR = 'data-theme-host'
export const THEME_HOST_REASON_ATTR = 'data-theme-host-reason'

/** The one notice the host shows, so a second one replaces the first. */
export const THEME_NOTICE_ID = 'theme-host-notice'

/** The frame drawn around the page; every theme registers one. */
export const FRAME_SLOT: SlotId = 'frame@1'

/** Every slot the host serves (registry-types.ts SlotId), each versioned with its props. */
export const THEME_SLOT_IDS: readonly SlotId[] = [
  'frame@1',
  'lookEditor@1',
  'styleSheet@1',
  'notFound@1',
  'asleep@1',
  'page-header.mascot@1',
  'page-header.art@1',
  'empty-state.mascot@1',
]

/**
 * The core pages a theme may register its own version of: Gerard's Q3 list (chat, deliverables,
 * missions, agents, knowledge, playbooks, command-center, home) on the app's routes (plan §2.6).
 */
export const THEME_PAGE_ROUTES: readonly string[] = [
  '/chat',
  '/deliverables',
  '/assignments',
  '/missions/[id]',
  '/agents',
  '/documents',
  '/playbooks',
  '/command-center',
  '/',
]

/** A theme route's path under /x/: up to four lower-case segments of letters, digits and hyphens. */
export const THEME_ROUTE_PATTERN = /^[a-z0-9][a-z0-9-]{0,63}(\/[a-z0-9][a-z0-9-]{0,63}){0,3}$/
