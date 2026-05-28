'use client'

/**
 * SummaryTab — Studio shell around the existing CommandCentreDashboard
 * widget grid.
 *
 * The user wants widgets to be flexible/resizable (drag-to-reorder, size
 * override 1/3 / 1/2 / 2/3 / full, show/hide via Customize, persisted to
 * localStorage). All of that already exists in CommandCentreDashboard;
 * rebuilding it would duplicate work and lose the customization state
 * users have already saved.
 *
 * Widget bodies were token-swept in Sprint 0.5 so they inherit the
 * Studio cream / olive / navy palette via the .studio scope.
 */

import dynamic from 'next/dynamic'
import { useRouter, useSearchParams, usePathname } from 'next/navigation'

const CommandCentreDashboard = dynamic(
  () =>
    import('@/components/activity/widgets/command-centre-dashboard').then(
      (m) => m.CommandCentreDashboard,
    ),
  { ssr: false, loading: () => <SummarySkeleton /> },
)

function SummarySkeleton() {
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-4">
        <div className="h-[320px] rounded-xl border border-border bg-card animate-pulse" />
        <div className="h-[320px] rounded-xl border border-border bg-card animate-pulse" />
      </div>
      <div className="h-[280px] rounded-xl border border-border bg-card animate-pulse" />
    </div>
  )
}

export function SummaryTab() {
  const router = useRouter()
  const pathname = usePathname() ?? '/command-center'
  const searchParams = useSearchParams()

  const goTab = (tab: string) => {
    const params = new URLSearchParams(searchParams?.toString() ?? '')
    params.set('tab', tab)
    router.push(`${pathname}?${params.toString()}` as any, { scroll: false })
  }

  return (
    <CommandCentreDashboard
      period="1d"
      onViewAllActivity={() => goTab('activity')}
      onViewCalendar={() => goTab('calendar')}
    />
  )
}
