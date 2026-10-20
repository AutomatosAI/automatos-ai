import { cookies } from 'next/headers'
import { redirect } from 'next/navigation'
import { findThemeEntry } from '@/lib/theme-host/registry'
import { CHAT_HREF } from '@/lib/ticket-links'
import { parseUiStyle, UI_STYLE_COOKIE } from '@/lib/ui-style'

/** The theme route a theme's Conversations page is served on (builtin Play's, PRD-260 K10). */
const CONVERSATIONS_ROUTE = 'conversations'
const THEME_CONVERSATIONS_HREF = `/x/${CONVERSATIONS_ROUTE}`

/**
 * PRD-260 §4, K10; PRD-261 A-04b — the old address of Play's Conversations page, which is now a
 * theme route at /x/conversations (app/x/[...path]/page.tsx). Under a theme whose registry entry
 * lists that route (builtin Play; the cookie `play` too), the server sends the browser there;
 * under Classic, Studio, any other theme and a missing or tampered cookie, to /chat (whose own
 * history lists the conversations), before anything paints. A-18 moves this into Play's shim.
 */
export default async function ConversationsPage(): Promise<never> {
  const store = await cookies()
  const entry = findThemeEntry(parseUiStyle(store.get(UI_STYLE_COOKIE)?.value))
  redirect(entry?.routes.includes(CONVERSATIONS_ROUTE) ? THEME_CONVERSATIONS_HREF : CHAT_HREF)
}
