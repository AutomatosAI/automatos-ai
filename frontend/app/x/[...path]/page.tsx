import { cookies } from 'next/headers'
import { notFound, redirect } from 'next/navigation'
import { AiLabelGate, type AiLabelPlacement } from '@/components/ai-label/ai-label'
import { MainLayout } from '@/components/layout/main-layout'
import { ThemeSlotRoute } from '@/components/theme-host/theme-slot-route'
import { VoiceProvider } from '@/components/voice/voice-provider'
import { THEME_ROUTE_PATTERN } from '@/lib/theme-host/constants'
import { isInstalledThemeStyle } from '@/lib/theme-host/registry'
import { CHAT_HREF } from '@/lib/ticket-links'
import { parseUiStyle, UI_STYLE_COOKIE } from '@/lib/ui-style'

/** Where a theme route's AI label sits: MainLayout's Auto widget takes the bottom right. */
const THEME_ROUTE_LABEL_PLACEMENT: AiLabelPlacement = 'bottom-left'

interface ThemeRoutePageProps {
  params: Promise<{ path: string[] }>
}

/**
 * PRD-261 A-04b — a theme's own pages, at /x/<route> (registerRoute). The server answers the 404
 * for an address no theme can declare (a route that fails THEME_ROUTE_PATTERN). Classic and Studio
 * serve no route, and no theme is compiled into the app (A-19f1) whose routes the server could know,
 * so under them the server sends the member to the chat: the address is a route a theme serves
 * (Play's /x/conversations, left open after a switch to Studio), never a page here. Under an
 * installed theme its routes are known only once its code loads in the browser, so the route
 * renders full bleed inside MainLayout (the theme's frame) through ThemeSlotRoute, which decides
 * there.
 *
 * PRD-264 EUAI-11a: the host's AI label gate wraps MainLayout, so the label sits outside the
 * theme's frame and route (installed Play's Conversations is a chat surface). Every theme route gets
 * it: a manifest's `uses` is checked by lib/theme-host/package-manifest.ts but not handed to the
 * host, so the host can't yet tell which routes use the `chat` area, and labels them all.
 *
 * Theme voice: the host voice provider sits inside the gate, around the theme's frame, so a theme's chat route and
 * its frame read the same voice session as the chat page does (app/chat/layout.tsx).
 */
export default async function ThemeRoutePage({ params }: ThemeRoutePageProps) {
  const { path } = await params
  const route = path.join('/')
  const store = await cookies()
  if (!THEME_ROUTE_PATTERN.test(route)) notFound()
  if (!isInstalledThemeStyle(parseUiStyle(store.get(UI_STYLE_COOKIE)?.value))) redirect(CHAT_HREF)
  return (
    <AiLabelGate placement={THEME_ROUTE_LABEL_PLACEMENT}>
      <VoiceProvider>
        <MainLayout fullBleed>
          <ThemeSlotRoute route={route} />
        </MainLayout>
      </VoiceProvider>
    </AiLabelGate>
  )
}
