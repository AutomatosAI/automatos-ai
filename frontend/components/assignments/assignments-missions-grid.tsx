'use client'

import { useRouter } from 'next/navigation'
import { Target } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { MissionList } from '@/components/missions/mission-list'

/**
 * Assignments → Missions tab grid (US-013).
 *
 * Thin wrapper around MissionList — identical pattern to
 * AssignmentsPlaybooksGrid wrapping PlaybooksTab.
 * MissionList already provides: filter chips, search, card grid,
 * loading skeletons, empty state, and create modal.
 */
export function AssignmentsMissionsGrid() {
  const router = useRouter()

  return (
    <MissionList
      emptyAction={
        <div className="flex items-center justify-center gap-3">
          <Button
            variant="secondary"
            size="sm"
            onClick={() => router.push('/chat?mode=plan&from=assignments')}
          >
            + Plan a Mission
          </Button>
        </div>
      }
    />
  )
}
