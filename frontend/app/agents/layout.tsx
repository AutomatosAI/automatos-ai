import type { ReactNode } from 'react'
import { ThemeSlot } from '@/components/theme-host/theme-slot'

/** The route a theme's agents page is registered on (registerPage, plan §2.6 Q3: agents). */
const AGENTS_ROUTE = '/agents'
/** Links into a surface only the core page has: another tab, an agent's details and its panel. */
const CORE_AGENTS_PARAMS = ['tab', 'agent', 'panel']
/** The tab a theme's agents page is: the roster, the core page's own default tab (lib/agents/tabs.ts). */
const THEME_SERVES = { tab: ['roster'] }

/**
 * PRD-261 A-04d — the Agents page through ThemeSlot. A theme that registers an /agents page draws
 * it instead at its width; a link into another tab or to one agent's details keeps the core page.
 * With no theme, or a theme that registers no /agents page, the core page renders unchanged.
 */
export default function AgentsLayout({ children }: { children: ReactNode }) {
  return (
    <ThemeSlot page={AGENTS_ROUTE} keepCorePage={CORE_AGENTS_PARAMS} themeServes={THEME_SERVES}>
      {children}
    </ThemeSlot>
  )
}
