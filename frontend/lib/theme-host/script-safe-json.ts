/**
 * PRD-261 A-03 (A0 §7 Q8) — JSON that can sit inside a <script> element, in a neutral home: the
 * import map (import-map.ts) and the theme boot element (components/theme-host/theme-boot.tsx)
 * both write it, and neither is Play's. Moved here from lib/looks/prepaint.ts.
 *
 * It escapes '<' (so no '</script>' or '<!--' can form), U+2028 and U+2029, which is enough inside
 * a script element. A caller writes its output as the whole element body, with nothing put
 * around it (A0 D16).
 */
export function scriptSafeJson(value: unknown): string {
  return JSON.stringify(value)
    .replace(/</g, '\\u003c')
    .replace(/ /g, '\\u2028')
    .replace(/ /g, '\\u2029')
}
