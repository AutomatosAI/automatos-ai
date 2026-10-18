import { cookies } from 'next/headers'
import { redirect } from 'next/navigation'
import { MainLayout } from '@/components/layout/main-layout'
import { PlayConversations } from '@/components/play/conversations/play-conversations'
import { BUILTIN_PLAY_STYLE } from '@/lib/theme-host/builtin/play-entry'
import { parseUiStyle, UI_STYLE_COOKIE } from '@/lib/ui-style'

/**
 * PRD-260 §4, K10 — Play's Conversations page. Only Play has one: under Classic
 * or Studio the server sends the browser to /chat (whose own history lists the
 * conversations) before anything paints. The style comes from the cookie
 * app/layout.tsx reads; a missing or tampered one is the default style. PRD-261
 * A-02b: the cookie parses to `theme:builtin:play` (the legacy `play` too), so
 * that is what keeps the page; A-04b moves it to /x/conversations.
 */
export default async function ConversationsPage() {
  const store = await cookies()
  if (parseUiStyle(store.get(UI_STYLE_COOKIE)?.value) !== BUILTIN_PLAY_STYLE) redirect('/chat')
  return (
    <MainLayout fullBleed>
      <PlayConversations />
    </MainLayout>
  )
}
