'use client'

import { useRouter } from 'next/navigation'
import { RefreshCw, Plus } from 'lucide-react'
import { toast } from 'react-hot-toast'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { RoutineCard } from './routine-card'
import { useHeartbeats } from '@/hooks/use-heartbeats-api'

// ─── Skeleton ────────────────────────────────────────────

function RoutineCardSkeleton() {
  return (
    <div className="glass-card p-4 space-y-3">
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-3">
          <Skeleton className="w-10 h-10 rounded-lg" />
          <div className="space-y-1.5">
            <Skeleton className="h-4 w-32" />
            <Skeleton className="h-3 w-20" />
          </div>
        </div>
        <Skeleton className="h-5 w-14 rounded-full" />
      </div>
      <Skeleton className="h-3 w-full" />
      <Skeleton className="h-3 w-3/4" />
      <div className="flex items-center gap-4">
        <Skeleton className="h-3 w-20" />
        <Skeleton className="h-3 w-24" />
        <Skeleton className="h-3 w-16" />
      </div>
      <Skeleton className="h-px w-full" />
      <div className="flex justify-end gap-2">
        <Skeleton className="h-8 w-16 rounded-md" />
        <Skeleton className="h-8 w-14 rounded-md" />
      </div>
    </div>
  )
}

// ─── Empty State ─────────────────────────────────────────

function RoutinesEmptyState() {
  const router = useRouter()

  return (
    <div className="glass-card p-8 text-center text-muted-foreground">
      <RefreshCw className="w-12 h-12 mx-auto mb-3 opacity-30" />
      <p className="font-medium">No routines set up</p>
      <p className="text-sm mt-1 max-w-md mx-auto">
        Routines let your agents check things automatically — like monitoring
        your inbox or tracking sales
      </p>
      <Button
        variant="outline"
        size="sm"
        className="mt-4"
        onClick={() => {
          router.push('/agents')
          toast('Select an agent to configure its routine', { icon: '🔄' })
        }}
      >
        Set Up a Routine
      </Button>
    </div>
  )
}

// ─── Component ───────────────────────────────────────────

export function ActivityRoutines() {
  const router = useRouter()
  const { data, isLoading } = useHeartbeats()

  const heartbeats = data?.heartbeats ?? []
  const hasHeartbeats = heartbeats.length > 0

  const handleNewRoutine = () => {
    router.push('/agents')
    toast('Select an agent to configure its routine', { icon: '🔄' })
  }

  if (isLoading) {
    return (
      <div className="space-y-4">
        <div className="flex items-center justify-end">
          <Skeleton className="h-9 w-36 rounded-md" />
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {Array.from({ length: 3 }).map((_, i) => (
            <RoutineCardSkeleton key={i} />
          ))}
        </div>
      </div>
    )
  }

  if (!hasHeartbeats) {
    return <RoutinesEmptyState />
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-end">
        <Button variant="outline" size="sm" onClick={handleNewRoutine}>
          <Plus className="w-4 h-4 mr-1.5" />
          New Routine
        </Button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {heartbeats.map((heartbeat, index) => (
          <RoutineCard
            key={heartbeat.id}
            heartbeat={heartbeat}
            animationDelay={index * 0.08}
          />
        ))}
      </div>
    </div>
  )
}
