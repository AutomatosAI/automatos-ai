import { cookies } from 'next/headers'
import { redirect } from 'next/navigation'
import { MainLayout } from '@/components/layout/main-layout'
import { PlayConversations } from '@/components/play/conversations/play-conversations'
import { parseUiStyle, UI_STYLE_COOKIE } from '@/lib/ui-style'

/**
 * PRD-260 §4, K10 — Play's Conversations page. Only Play has one: under Classic
 * or Studio the server sends the browser to /chat (whose own history lists the
 * conversations) before anything paints. The style comes from the cookie
 * app/layout.tsx reads; a missing or tampered one is the default style.
 */
export default async function ConversationsPage() {
  const store = await cookies()
  if (parseUiStyle(store.get(UI_STYLE_COOKIE)?.value) !== 'play') redirect('/chat')
  return (
    <MainLayout fullBleed>
      <PlayConversations />
    </MainLayout>
  )
}
