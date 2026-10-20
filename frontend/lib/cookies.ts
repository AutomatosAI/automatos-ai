/**
 * The one cookie writer (PRD-260 P260-FX-27, M88). The style cookie
 * (contexts/ui-style-context.tsx) and Play's tone cookie (written by the host,
 * lib/theme-host/builtin/play-tone-cookie.ts) are read by app/layout.tsx on
 * the next request, so they carry the same flags: the whole site, Lax, Secure
 * when the page is on https. A Max-Age of 0 deletes the cookie.
 */
export function writeCookie(name: string, value: string, maxAgeS: number): void {
  const secure = window.location.protocol === 'https:' ? '; Secure' : ''
  document.cookie = `${name}=${value}; Path=/; Max-Age=${maxAgeS}; SameSite=Lax${secure}`
}
