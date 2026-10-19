import type { ReactNode } from 'react'
import { ThemeSlot } from '@/components/theme-host/theme-slot'

/** The route a theme's Command Center page is registered on (registerPage, plan §2.6 Q3: command-center). */
const COMMAND_CENTER_ROUTE = '/command-center'
/**
 * Links into a surface only the core page has: another tab (Board, Questions, …), a Task on the
 * board (task_id: BoardTab's useTicketDeepLink, both bases), and one execution or Playbook
 * (openExecution, recipeId). A theme's base may be Classic (THEME_BASES), and then the core page
 * under it is components/activity/activity-page.tsx, which opens those in its feed. agent_id is not
 * listed: nothing reads it, and every link that sets it also sets tab=board.
 */
const CORE_COMMAND_CENTER_PARAMS = ['tab', 'openExecution', 'recipeId', 'task_id']
/** The tab a theme's page is: Summary, the core page's own default tab (command-center-shell.tsx). */
const THEME_SERVES = { tab: ['summary'] }

/**
 * PRD-261 A-04d — the Command Center through ThemeSlot. A theme that registers a /command-center
 * page draws it instead at its width; a link into another tab or one of the deep links keeps the
 * core page. With no theme, or a theme that registers no /command-center page, the core page
 * renders unchanged.
 */
export default function CommandCenterLayout({ children }: { children: ReactNode }) {
  return (
    <ThemeSlot page={COMMAND_CENTER_ROUTE} keepCorePage={CORE_COMMAND_CENTER_PARAMS} themeServes={THEME_SERVES}>
      {children}
    </ThemeSlot>
  )
}
