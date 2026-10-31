'use client'

/**
 * PRD-261 C-03b — `/admin/themes`, the review queue for Marketplace theme versions.
 *
 * Both editions (not in SAAS_ONLY_ROUTES): the local operator is the instance's super admin and
 * reviews the themes submitted on their own machine. The queue itself is
 * components/admin/themes/review-queue.tsx; the server checks `caller_is_admin` on every route.
 */
import { ThemeReviewQueue } from '@/components/admin/themes/review-queue'
import { MainLayout } from '@/components/layout/main-layout'
import { PageHeader } from '@/components/shared'

export default function AdminThemesPage() {
  return (
    <MainLayout>
      <div className="space-y-6">
        <PageHeader
          title="Theme"
          titleAccent="Review Queue"
          eyebrow="Marketplace · theme review"
          lede="Read each submitted version's scan report, then approve or reject it. A person reviews every version."
        />
        <ThemeReviewQueue />
      </div>
    </MainLayout>
  )
}
