import { Suspense, type ReactNode } from 'react'
import { DeliverableDeepLink } from '@/components/deliverables/deliverable-deep-link'
import { ThemeSlot } from '@/components/theme-host/theme-slot'
import type { DeliverableTab } from '@/lib/deliverables/tabs'
import { DELIVERABLES_HREF } from '@/lib/ticket-links'

/** Links into one of the page's other sections: a tab (Blogs, Templates, Brand kit, Socials), a post, a type drill-down. */
const CORE_DELIVERABLES_PARAMS = ['tab', 'post', 'artifact_type']
/**
 * The tab a theme's Deliverables page is: the core's Outputs feed (installed Play's shelf, for
 * one; PRD-260 K8). Every `?tab=outputs` link (the explorer's Back, Generate's 'Open Deliverables', the
 * backend's own `?tab=outputs&deliverable=<id>`) keeps the theme's page, and the page opens the
 * Deliverable it names; a post or a type drill-down still keeps the core page.
 */
const THEME_PAGE_TAB: DeliverableTab = 'outputs'
const THEME_SERVES = { tab: [THEME_PAGE_TAB] }

/**
 * The Deliverables pages, with the one a link names opened over them (F298). A theme that
 * registers a /deliverables page (installed Play, for one: its shelf) draws it instead at its
 * width, through ThemeSlot (PRD-261 A-04b); not the explorer. The theme's page opens a linked
 * Deliverable as its own page (K8), so the deep link's sheet mounts beside the core page only:
 * one detail, never two.
 */
export default function DeliverablesLayout({ children }: { children: ReactNode }) {
  return (
    <ThemeSlot page={DELIVERABLES_HREF} keepCorePage={CORE_DELIVERABLES_PARAMS} themeServes={THEME_SERVES}>
      {children}
      <Suspense fallback={null}>
        <DeliverableDeepLink />
      </Suspense>
    </ThemeSlot>
  )
}
