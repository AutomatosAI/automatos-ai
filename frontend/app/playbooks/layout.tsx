import type { ReactNode } from 'react'
import { ThemeSlot } from '@/components/theme-host/theme-slot'

/** The route a theme's playbooks page is registered on (registerPage, plan §2.6 Q3: playbooks). */
const PLAYBOOKS_ROUTE = '/playbooks'
/** A link to one Playbook's run (F242: the library's Run, ?id=), which only the core's dialog opens. */
const CORE_PLAYBOOKS_PARAMS = ['id']

/**
 * PRD-261 A-04d — the Playbooks page through ThemeSlot. A theme that registers a /playbooks page
 * draws it instead at its width, so the core page (and its forward to the Assignments hub under
 * Studio) never mounts; a link to one Playbook's run keeps the core page. With no theme, or a
 * theme that registers no /playbooks page, the core page renders unchanged.
 */
export default function PlaybooksLayout({ children }: { children: ReactNode }) {
  return (
    <ThemeSlot page={PLAYBOOKS_ROUTE} keepCorePage={CORE_PLAYBOOKS_PARAMS}>
      {children}
    </ThemeSlot>
  )
}
