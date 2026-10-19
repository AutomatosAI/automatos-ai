/**
 * PRD-261 A-12a — `cards.CorePage(children)`: core's own page inside a theme's page.
 *
 * A theme page drawn on a core route (registerPage) can keep a surface only the core page has, a
 * tab the SDK v1 has no data hooks for (docs/theme-host/sdk-surface-map.md §3), by rendering
 * `<CorePage />` where it wants it. The host hands the core page down through CorePageProvider,
 * around the theme's version of the page; where there is no core page (a theme route under /x/,
 * or outside the provider) CorePage renders its `children` instead.
 *
 * This module imports React only: the host's page seam (components/theme-host/theme-slot.tsx)
 * may load it with every page without pulling the SDK's cards along (PRD-260 M17's seam graph).
 * CorePageProvider is host-only: it is not on the SDK surface (surface/cards.json).
 */
import { createContext, useContext, useMemo, type ReactElement, type ReactNode } from 'react'

export interface CorePageProps {
  /** What to render where there is no core page. */
  readonly children?: ReactNode
}

export interface CorePageProviderProps {
  /** The core's own page for this route. */
  readonly page: ReactNode
  /** The theme's version of the page. */
  readonly children: ReactNode
}

interface CorePageValue {
  readonly page: ReactNode
}

const CorePageContext = createContext<CorePageValue | null>(null)

/** Host-only: hands the core page down to the theme's version of it. */
export function CorePageProvider({ page, children }: CorePageProviderProps): ReactElement {
  const value = useMemo<CorePageValue>(() => ({ page }), [page])
  return <CorePageContext.Provider value={value}>{children}</CorePageContext.Provider>
}

/** Core's own page for this route, or `children` where there is none. */
export function CorePage({ children }: CorePageProps): ReactElement {
  const core = useContext(CorePageContext)
  return <>{core === null ? children : core.page}</>
}
