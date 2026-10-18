import { THEME_BOOT_ELEMENT_ID, THEME_BOOT_SCRIPT, type ThemeBoot } from '@/lib/theme-host/boot'
import { scriptSafeJson } from '@/lib/theme-host/script-safe-json'

/** The JSON element's type: the browser never runs it, THEME_BOOT_SCRIPT reads it. */
const BOOT_DATA_TYPE = 'application/json'

interface ThemeBootScriptsProps {
  /** Already resolved and checked (lib/theme-host/boot.ts, sanitizeThemeBoot). */
  readonly boot: ThemeBoot
}

/**
 * PRD-261 A-03 — a server component: the style's first-paint data as one JSON element, then the
 * one constant script that applies it, both in <head> right after the import map. The element's
 * body is scriptSafeJson(boot) with nothing concatenated (A0 D16), so no '</script>' or '<!--'
 * can form in it; the script is a literal, the same for every style and every user.
 */
export function ThemeBootScripts({ boot }: ThemeBootScriptsProps) {
  return (
    <>
      <script type={BOOT_DATA_TYPE} id={THEME_BOOT_ELEMENT_ID} dangerouslySetInnerHTML={{ __html: scriptSafeJson(boot) }} />
      <script dangerouslySetInnerHTML={{ __html: THEME_BOOT_SCRIPT }} />
    </>
  )
}
