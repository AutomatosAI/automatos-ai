import type { ReactNode } from 'react'
import { ThemeSlot } from '@/components/theme-host/theme-slot'

/** The route a theme's missions page is registered on (registerPage, plan §2.6 Q3: missions, D0). */
const ASSIGNMENTS_ROUTE = '/assignments'
/** Links into a surface only the core page has: another tab (Playbooks) and a Playbook's run dialog (F242). */
const CORE_ASSIGNMENTS_PARAMS = ['tab', 'id']
/** The tab a theme's page is: Missions, where /missions forwards (app/missions/page.tsx). */
const THEME_SERVES = { tab: ['missions'] }

/**
 * PRD-261 A-04d — the Assignments page through ThemeSlot. Gerard's Q3 "missions" is this route's
 * Missions tab and /missions/[id] (D0), so a theme that registers an /assignments page draws it
 * instead at its width, for /assignments and every /missions link (its filters too, ?state=);
 * the Playbooks tab and a Playbook's run keep the core page. With no theme, or a theme that
 * registers no /assignments page, the core page renders unchanged.
 */
export default function AssignmentsLayout({ children }: { children: ReactNode }) {
  return (
    <ThemeSlot page={ASSIGNMENTS_ROUTE} keepCorePage={CORE_ASSIGNMENTS_PARAMS} themeServes={THEME_SERVES}>
      {children}
    </ThemeSlot>
  )
}
