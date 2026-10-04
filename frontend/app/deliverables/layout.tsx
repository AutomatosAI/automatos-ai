import { Suspense, type ReactNode } from 'react'
import { DeliverableDeepLink } from '@/components/deliverables/deliverable-deep-link'

/** The Deliverables pages, with the one a link names opened over them (F298). */
export default function DeliverablesLayout({ children }: { children: ReactNode }) {
  return (
    <>
      {children}
      <Suspense fallback={null}>
        <DeliverableDeepLink />
      </Suspense>
    </>
  )
}
