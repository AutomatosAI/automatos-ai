import { Suspense, type ReactNode } from 'react'
import { DeliverableDeepLink } from '@/components/deliverables/deliverable-deep-link'
import { PlayRouteSwitch } from '@/components/play/play-route-switch'
import { PlayShelf } from '@/components/play/deliverables/play-shelf'

/** The Deliverables page under Play at >= 1024 px: Play's shelf (or one Deliverable) instead of the page (PRD-260 K7, K8). Not the explorer. */
const PLAY_DELIVERABLES_PATHS = ['/deliverables']
/** Links into one of the page's other sections: a tab (Blogs, Templates, Brand kit, Socials), a post, a type drill-down. */
const CORE_DELIVERABLES_PARAMS = ['tab', 'post', 'artifact_type']

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
