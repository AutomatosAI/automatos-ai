import { cookies } from 'next/headers'
import { notFound } from 'next/navigation'
import { MainLayout } from '@/components/layout/main-layout'
import { ThemeSlotRoute } from '@/components/theme-host/theme-slot-route'
import { THEME_ROUTE_PATTERN } from '@/lib/theme-host/constants'
import { findThemeEntry, isInstalledThemeStyle } from '@/lib/theme-host/registry'
import { parseUiStyle, UI_STYLE_COOKIE, type UiStyle } from '@/lib/ui-style'

interface ThemeRoutePageProps {
  params: Promise<{ path: string[] }>
}

/**
 * Whether the style's theme may serve `route`: a builtin theme serves the routes its registry
 * entry lists; an installed theme's routes are known only once its code loads in the browser,
 * so ThemeSlotRoute decides there. Classic and Studio serve none.
 */
function themeMayServe(style: UiStyle, route: string): boolean {
  if (!THEME_ROUTE_PATTERN.test(route)) return false
  if (isInstalledThemeStyle(style)) return true
  return findThemeEntry(style)?.routes.includes(route) ?? false
}

/**
 * PRD-261 A-04b — a theme's own pages, at /x/<route> (registerRoute). The server answers the 404
 * for an address no theme of this style can serve (a route that fails THEME_ROUTE_PATTERN, Classic,
 * Studio, a builtin theme without the route), so those never reach the browser as a page; otherwise
 * the route renders full bleed inside MainLayout (the theme's frame) through ThemeSlotRoute.
 */
export default async function ThemeRoutePage({ params }: ThemeRoutePageProps) {
  const { path } = await params
  const route = path.join('/')
  const store = await cookies()
  if (!themeMayServe(parseUiStyle(store.get(UI_STYLE_COOKIE)?.value), route)) notFound()
  return (
    <MainLayout fullBleed>
      <ThemeSlotRoute route={route} />
    </MainLayout>
  )
}
