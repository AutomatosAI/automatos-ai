import { THEME_BOOT_ELEMENT_ID, THEME_BOOT_SCRIPT, type ThemeBoot } from '@/lib/theme-host/boot'
import { scriptSafeJson } from '@/lib/theme-host/script-safe-json'

/** Cookies a test hands the boot resolver. */
export function cookieReader(cookies: Readonly<Record<string, string>> = {}): (name: string) => string | undefined {
  return (name) => (Object.prototype.hasOwnProperty.call(cookies, name) ? cookies[name] : undefined)
}

/**
 * Runs the boot the way a page does: the JSON element as the root layout writes it, then the
 * constant script as a classic script. Only in tests: the app itself never evaluates strings.
 * `body` overrides the element's text (a malformed or hostile boot).
 */
export function runThemeBoot(boot: ThemeBoot | null, body: string = scriptSafeJson(boot)): void {
  const element = document.createElement('script')
  element.type = 'application/json'
  element.id = THEME_BOOT_ELEMENT_ID
  element.textContent = body
  document.head.appendChild(element)
  try {
    new Function(THEME_BOOT_SCRIPT)()
  } finally {
    element.remove()
  }
}

/** Runs the constant script with no boot element on the page. */
export function runThemeBootScriptAlone(): void {
  new Function(THEME_BOOT_SCRIPT)()
}
