import { Suspense, type ReactNode } from 'react'
import { DeliverableDeepLink } from '@/components/deliverables/deliverable-deep-link'
import { PlayRouteSwitch } from '@/components/play/play-route-switch'
import { PlayShelf } from '@/components/play/deliverables/play-shelf-lazy'
import type { DeliverableTab } from '@/lib/deliverables/tabs'

/** The Deliverables page under Play at >= 1024 px: Play's shelf (or one Deliverable) instead of the page (PRD-260 K7, K8). Not the explorer. */
const PLAY_DELIVERABLES_PATHS = ['/deliverables']
/** Links into one of the page's other sections: a tab (Blogs, Templates, Brand kit, Socials), a post, a type drill-down. */
const CORE_DELIVERABLES_PARAMS = ['tab', 'post', 'artifact_type']
/**
 * The tab Play's shelf is: the core's Outputs feed. Every `?tab=outputs` link
 * (the explorer's Back, Generate's 'Open Deliverables', the backend's own
 * `?tab=outputs&deliverable=<id>`) keeps Play's screen, and the shelf opens the
 * Deliverable it names (K8); a post or a type drill-down still keeps the page.
 */
const PLAY_SHELF_TAB: DeliverableTab = 'outputs'
const PLAY_SERVES = { tab: [PLAY_SHELF_TAB] }

/**
 * The Deliverables pages, with the one a link names opened over them (F298).
 * Under Play the shelf switch opens it as Play's own page (K8), so the deep
 * link's sheet mounts beside the core's page only: one detail, never two.
 */
export default function DeliverablesLayout({ children }: { children: ReactNode }) {
  return (
    <PlayRouteSwitch
      paths={PLAY_DELIVERABLES_PATHS}
      keepPageFor={CORE_DELIVERABLES_PARAMS}
      playServes={PLAY_SERVES}
      play={
        <Suspense fallback={null}>
          <PlayShelf />
        </Suspense>
      }
    >
      {children}
      <Suspense fallback={null}>
        <DeliverableDeepLink />
      </Suspense>
    </PlayRouteSwitch>
  )
}
