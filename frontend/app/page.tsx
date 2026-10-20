import { cookies } from 'next/headers'
import { redirect } from 'next/navigation'

import { CoreHomeForward } from '@/components/theme-host/theme-slot-home'
import { ThemeSlot } from '@/components/theme-host/theme-slot'
import { findThemeEntry, isInstalledThemeStyle } from '@/lib/theme-host/registry'
import { CHAT_HREF } from '@/lib/ticket-links'
import { parseUiStyle, UI_STYLE_COOKIE, type UiStyle } from '@/lib/ui-style'

/** The route a theme's home page is registered on (registerPage, plan §2.6 Q3: home, D0). */
const HOME_ROUTE = '/'

/**
 * Whether the style's theme may draw a home page: an installed theme's pages are known only once
 * its code loads in the browser, so ThemeSlot decides there; a builtin theme draws one when its
 * registry entry owns `/`. Classic and Studio draw none.
 */
function themeMayOwnHome(style: UiStyle): boolean {
  if (isInstalledThemeStyle(style)) return true
  return findThemeEntry(style)?.ownsPages.includes(HOME_ROUTE) ?? false
}

/**
 * The home page. The core has none: it forwards to Chat, on the server, before anything paints,
 * whenever no theme of the style can draw one. PRD-261 A-04d: under a style whose theme may (an
 * installed theme), a theme's registered `/` page draws through ThemeSlot instead, and without
 * one the browser forwards to Chat (CoreHomeForward).
 */
export default async function Home() {
  const store = await cookies()
  if (!themeMayOwnHome(parseUiStyle(store.get(UI_STYLE_COOKIE)?.value))) redirect(CHAT_HREF)
  return (
    <ThemeSlot page={HOME_ROUTE}>
      <CoreHomeForward />
    </ThemeSlot>
  )
}
