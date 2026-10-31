import { THEME_BOOT_ELEMENT_ID } from '../boot'
import { sanitizeThemeBoot } from '../boot-sanitize'
import type { ThemeBoot } from '../boot-types'
import { themeLookRulesOf, type ThemeLookRules } from './spec'

/**
 * PRD-261 A-19d2 (plan §2.2.4) — an installed theme's looks as the page carries them: the root
 * layout writes the style's boot into the page as one JSON element (components/theme-host/
 * theme-boot.tsx), its prepaint table, look guard and presets among it (A-19c), so the generic look
 * runtime runs the theme's looks from the same data the boot script painted the first frame with.
 * The element is read in the browser only, after mount, and passes the browser's boot gate again
 * (boot-sanitize.ts): it is page text, so it is checked like anything else from outside the code.
 *
 * The looks are the theme's only when the boot's entry names that theme: a boot for another style
 * (a server re-render after the style changed in place) has none. A switch to an installed theme
 * always reloads the page (PA-13), so the boot the page loaded with is the running theme's. A base
 * boot (the server's boot fetch timed out, was refused or failed) names no theme at all: the look
 * runtime then reads the rules from the install's boot the browser fetches (../boot-client.ts).
 */

const LOG_PREFIX = '[looks]'

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

/** The page's boot, through the gate; null when there is no element or it isn't a boot. */
export function readPageBoot(): ThemeBoot | null {
  try {
    const raw: unknown = JSON.parse(document.getElementById(THEME_BOOT_ELEMENT_ID)?.textContent || 'null')
    return isRecord(raw) ? sanitizeThemeBoot(raw as unknown as ThemeBoot) : null
  } catch (error) {
    console.warn(`${LOG_PREFIX} the page's theme boot couldn't be read, so the theme's looks don't run:`, error)
    return null
  }
}

/** Whether a boot names a running theme (it carries the theme's entry); a base boot never does. */
export function bootNamesTheme(boot: ThemeBoot | null): boolean {
  return boot !== null && boot.entry !== null
}

/** The look rules of `themeId` in `boot`; null when the boot is another theme's, a base boot, or has no looks. */
export function bootLookRules(boot: ThemeBoot | null, themeId: string): ThemeLookRules | null {
  if (boot === null || boot.entry === null || boot.entry.themeId !== themeId) return null
  return themeLookRulesOf(boot)
}
