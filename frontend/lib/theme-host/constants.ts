/**
 * PRD-261 A-01 — the theme-host spike's fixed values.
 *
 * The spike's one theme lives at a fixed same-origin path, so no theme URL ever comes from a
 * user. Nothing under frontend/ serves that path: only the browser lane's route interception
 * does, so a spike build without it falls back to Studio.
 */
import { STYLE_HTML_CLASS_NAMES, type UiStyle } from '@/lib/ui-style'

export const THEME_SPIKE_ROOT = '/theme-spike/hello/'
export const THEME_SPIKE_MANIFEST_PATH = '/theme-spike/hello/theme.json'

/** How long the entry module and the stylesheet may take before the load counts as failed. */
export const THEME_LOAD_TIMEOUT_MS = 10_000

/** `?safe_mode=themes` turns theme loading off for this tab (PRD-261 §6). */
export const SAFE_MODE_PARAM = 'safe_mode'
export const SAFE_MODE_VALUE = 'themes'
export const SAFE_MODE_STORAGE_KEY = 'automatos-theme-safe-mode'

/** A theme's own class on <html>: 3 to 32 characters, lower-case letters, digits and hyphens. */
export const HTML_CLASS_PATTERN = /^[a-z][a-z0-9-]{2,31}$/

/** The style the spike's theme is drawn on, and the only style it loads under. */
export const SPIKE_BASE_STYLE: UiStyle = 'studio'

/** Classes the core owns on <html> (every Style-axis class, and the tone's), never a theme's. */
export const RESERVED_HTML_CLASSES: readonly string[] = [...STYLE_HTML_CLASS_NAMES, 'classic', 'dark', 'light']

/** Mirrors of the host's status and failure reason on <html>, for the browser lane to wait on. */
export const THEME_HOST_STATUS_ATTR = 'data-theme-host'
export const THEME_HOST_REASON_ATTR = 'data-theme-host-reason'

/** The one notice the host shows, so a second one replaces the first. */
export const THEME_NOTICE_ID = 'theme-host-notice'

/** The spike's only slot: the frame drawn around the page. */
export const FRAME_SLOT = 'frame@1'
