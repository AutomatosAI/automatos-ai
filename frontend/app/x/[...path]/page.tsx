import { cookies } from 'next/headers'
import { notFound } from 'next/navigation'
import { AiLabelGate, type AiLabelPlacement } from '@/components/ai-label/ai-label'
import { MainLayout } from '@/components/layout/main-layout'
import { ThemeSlotRoute } from '@/components/theme-host/theme-slot-route'
import { THEME_ROUTE_PATTERN } from '@/lib/theme-host/constants'
import { isInstalledThemeStyle } from '@/lib/theme-host/registry'
import { parseUiStyle, UI_STYLE_COOKIE, type UiStyle } from '@/lib/ui-style'

/** Where a theme route's AI label sits: MainLayout's Auto widget takes the bottom right. */
const THEME_ROUTE_LABEL_PLACEMENT: AiLabelPlacement = 'bottom-left'

interface ThemeRoutePageProps {
  params: Promise<{ path: string[] }>
}

/**
 * Whether the style's theme may serve `route`: an installed theme's routes are known only once its
 * code loads in the browser, so ThemeSlotRoute decides there. Classic and Studio serve none, and no
 * theme is compiled into the app (A-19f1) whose routes the server could know.
 */
function themeMayServe(style: UiStyle, route: string): boolean {
  return THEME_ROUTE_PATTERN.test(route) && isInstalledThemeStyle(style)
}

/**
 * PRD-261 A-04b — a theme's own pages, at /x/<route> (registerRoute). The server answers the 404
 * for an address no theme of this style can serve (a route that fails THEME_ROUTE_PATTERN, Classic,
 * Studio), so those never reach the browser as a page; otherwise
 * the route renders full bleed inside MainLayout (the theme's frame) through ThemeSlotRoute.
 *
 * PRD-264 EUAI-11a: the host's AI label gate wraps MainLayout, so the label sits outside the
 * theme's frame and route (installed Play's Conversations is a chat surface). Every theme route gets
 * it: a manifest's `uses` is checked by lib/theme-host/package-manifest.ts but not handed to the
 * host, so the host can't yet tell which routes use the `chat` area, and labels them all.
 */
export default async function ThemeRoutePage({ params }: ThemeRoutePageProps) {
  const { path } = await params
  const route = path.join('/')
  const store = await cookies()
  if (!themeMayServe(parseUiStyle(store.get(UI_STYLE_COOKIE)?.value), route)) notFound()
  return (
    <AiLabelGate placement={THEME_ROUTE_LABEL_PLACEMENT}>
      <MainLayout fullBleed>
        <ThemeSlotRoute route={route} />
      </MainLayout>
    </AiLabelGate>
  )
}
