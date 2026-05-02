'use client'

import { useCallback, useMemo, useState } from 'react'
import { useSearchParams, useRouter } from 'next/navigation'
import { motion } from 'framer-motion'
import {
  ClipboardList,
  BookOpen,
  Rocket,
  Sparkles,
  Lightbulb,
  Zap,
  Download,
  Play,
} from 'lucide-react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Card, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { CreateMissionModal } from '@/components/missions/create-mission-modal'
import { CreatePlaybookModal } from '@/components/workflows/create-playbook-modal'
import { CreateTaskDialog } from '@/components/activity/board/create-task-dialog'
import { AssignmentsPlaybooksGrid } from '@/components/assignments/assignments-playbooks-grid'
import { AssignmentsMissionsGrid } from '@/components/assignments/assignments-missions-grid'
import { MissionConstellationCard } from '@/components/assignments/mission-card-constellation'
import { useWorkflowPlaybooks } from '@/hooks/use-playbook-api'
import { useMissions } from '@/hooks/use-missions-api'
import { useRecommendedAssignments } from '@/hooks/use-assignments-api'
import type { RecommendedItem } from '@/hooks/use-assignments-api'

// ── Types ──────────────────────────────────────────────────────────

type AssignmentTab = 'playbooks' | 'missions'

const VALID_TABS: ReadonlyArray<AssignmentTab> = ['playbooks', 'missions']

function resolveTab(param: string | null): AssignmentTab {
  if (param && VALID_TABS.includes(param as AssignmentTab)) return param as AssignmentTab
  return 'playbooks'
}

// ── Featured Hero Cards (US-010) ─────────────────────────────────

function FeaturedHeroCards() {
  const router = useRouter()
  const [missionOpen, setMissionOpen] = useState(false)
  const [playbookOpen, setPlaybookOpen] = useState(false)
  const [taskOpen, setTaskOpen] = useState(false)

  return (
    <div>
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Mission — Constellation hero card (large, top-left) */}
        <MissionConstellationCard onStart={() => setMissionOpen(true)} />

        {/* Right column: Playbook + Plan + Task stacked */}
        <div className="flex flex-col gap-3">
          {/* Playbook card (medium) */}
          <Card
            role="button"
            tabIndex={0}
            onClick={() => setPlaybookOpen(true)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault()
                setPlaybookOpen(true)
              }
            }}
            className="cursor-pointer overflow-hidden border-border/50 bg-card/60 backdrop-blur-sm hover:border-primary/30 hover:bg-card/80 transition-all group focus:outline-none focus:ring-2 focus:ring-primary/40"
          >
            <CardContent className="p-4 flex items-start gap-3">
              <div className="w-10 h-10 rounded-xl bg-primary/15 border border-primary/20 flex items-center justify-center shrink-0 group-hover:bg-primary/25 transition-colors">
                <BookOpen className="h-5 w-5 text-primary" />
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between mb-0.5">
                  <h3 className="font-semibold text-sm">Playbook</h3>
                  <span className="text-xs text-muted-foreground/70 group-hover:text-primary transition-colors">+</span>
                </div>
                <p className="text-xs text-muted-foreground leading-snug">
                  Reusable routines. Schedulable, triggerable.
                </p>
              </div>
            </CardContent>
          </Card>

          {/* Plan card (medium) */}
          <Card
            role="button"
            tabIndex={0}
            onClick={() => router.push('/chat?mode=plan&from=assignments')}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault()
                router.push('/chat?mode=plan&from=assignments')
              }
            }}
            className="cursor-pointer overflow-hidden border-border/50 bg-card/60 backdrop-blur-sm hover:border-[hsl(var(--warning))]/30 hover:bg-card/80 transition-all group focus:outline-none focus:ring-2 focus:ring-[hsl(var(--warning))]/40"
          >
            <CardContent className="p-4 flex items-start gap-3">
              <div className="w-10 h-10 rounded-xl bg-[hsl(var(--warning))]/15 border border-[hsl(var(--warning))]/20 flex items-center justify-center shrink-0 group-hover:bg-[hsl(var(--warning))]/25 transition-colors">
                <Lightbulb className="h-5 w-5 text-[hsl(var(--warning))]" />
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between mb-0.5">
                  <h3 className="font-semibold text-sm">Plan</h3>
                  <span className="text-xs text-muted-foreground/70 group-hover:text-[hsl(var(--warning))] transition-colors">+</span>
                </div>
                <p className="text-xs text-muted-foreground leading-snug">
                  Iterate with Auto. Refine before launching.
                </p>
              </div>
            </CardContent>
          </Card>

          {/* Task card (small) */}
          <Card
            role="button"
            tabIndex={0}
            onClick={() => setTaskOpen(true)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault()
                setTaskOpen(true)
              }
            }}
            className="cursor-pointer overflow-hidden border-border/50 bg-card/60 backdrop-blur-sm hover:border-cyan-500/30 hover:bg-card/80 transition-all group focus:outline-none focus:ring-2 focus:ring-cyan-500/40"
          >
            <CardContent className="p-3 flex items-center gap-3">
              <div className="w-9 h-9 rounded-lg bg-cyan-500/15 border border-cyan-500/20 flex items-center justify-center shrink-0 group-hover:bg-cyan-500/25 transition-colors">
                <Zap className="h-4 w-4 text-cyan-400" />
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-semibold">Task</h3>
                  <span className="text-xs text-muted-foreground/70 group-hover:text-cyan-400 transition-colors">+</span>
                </div>
                <p className="text-xs text-muted-foreground leading-snug">Quick single action.</p>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Create modals */}
      <CreateMissionModal open={missionOpen} onOpenChange={setMissionOpen} />
      <CreatePlaybookModal open={playbookOpen} onClose={() => setPlaybookOpen(false)} />
      <CreateTaskDialog open={taskOpen} onOpenChange={setTaskOpen} />
    </div>
  )
}

// ── Contextual Hero Hint (US-011) ──────────────────────────────────

const TWENTY_FOUR_HOURS_MS = 24 * 60 * 60 * 1000

function useHintMessage(): string | null {
  const { data: playbookData } = useWorkflowPlaybooks({ limit: 1 })
  const { data: missionData } = useMissions({ limit: 1 })
  const { data: failedData } = useMissions({ state: 'failed', limit: 5 })

  return useMemo(() => {
    const playbookCount = ((playbookData as any)?.items ?? []).length
    const missionCount = missionData?.missions?.length ?? 0

    // Check for recent failed runs in last 24h
    const now = Date.now()
    const recentFailed = (failedData?.missions ?? []).some((m) => {
      const ts = m.updated_at ?? m.created_at
      return ts && now - new Date(ts).getTime() < TWENTY_FOUR_HOURS_MS
    })

    if (recentFailed) return 'Something not working? Re-plan it with Auto.'
    if (playbookCount === 0 && missionCount === 0) return 'Not sure where to start? Plan it with Auto.'
    if (playbookCount > 0 && missionCount === 0) return 'Ready for something bigger? Try a Mission.'
    if (missionCount > 0 && playbookCount > 0) return 'What\u2019s next? Browse the Marketplace.'

    return null
  }, [playbookData, missionData, failedData])
}

function ContextualHeroHint() {
  const hint = useHintMessage()

  if (!hint) return null

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.15 }}
    >
      <p className="text-sm text-muted-foreground">{hint}</p>
    </motion.div>
  )
}

// ── Recommended Grid (US-016) ──────────────────────────────────────

function RecommendedCard({
  item,
  index,
  onClick,
}: {
  item: RecommendedItem
  index: number
  onClick: () => void
}) {
  const isMarketplace = item.source === 'marketplace'
  const isMission = item.type === 'mission'
  const Icon = isMission ? Rocket : BookOpen
  const isNew = (() => {
    if (!item.created_at) return false
    const ageMs = Date.now() - new Date(item.created_at).getTime()
    return ageMs < 14 * 24 * 60 * 60 * 1000
  })()
  const isFeatured = !!item.is_featured

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: index * 0.04 }}
      className="snap-start shrink-0"
    >
      <Card
        role="button"
        tabIndex={0}
        className="cursor-pointer w-[260px] h-[148px] flex flex-col overflow-hidden border-border/50 bg-card/60 backdrop-blur-sm hover:border-primary/30 hover:bg-card/80 transition-all focus:outline-none focus:ring-2 focus:ring-primary/40 group"
        onClick={onClick}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault()
            onClick()
          }
        }}
      >
        <CardContent className="p-3.5 flex-1 flex flex-col gap-2">
          <div className="flex items-start gap-2.5">
            <div
              className={`w-9 h-9 rounded-lg border flex items-center justify-center shrink-0 transition-colors ${
                isMission
                  ? 'bg-primary/15 border-primary/25 group-hover:bg-primary/25'
                  : 'bg-primary/10 border-primary/20 group-hover:bg-primary/20'
              }`}
            >
              <Icon className="h-4 w-4 text-primary" />
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-start gap-1.5">
                <h3 className="font-semibold text-sm leading-tight line-clamp-1 flex-1">
                  {item.name}
                </h3>
                {isFeatured ? (
                  <span className="font-mono text-[9px] tracking-wider px-1.5 py-px rounded bg-primary/15 text-primary border border-primary/30 shrink-0">
                    FEATURED
                  </span>
                ) : isNew ? (
                  <span className="font-mono text-[9px] tracking-wider px-1.5 py-px rounded bg-success/15 text-success border border-success/30 shrink-0">
                    NEW
                  </span>
                ) : null}
              </div>
              <p className="text-[11px] text-muted-foreground line-clamp-2 leading-snug mt-0.5">
                {item.description || 'No description'}
              </p>
            </div>
          </div>
        </CardContent>
        <div className="px-3.5 pb-2.5 pt-1 flex items-center justify-between text-[10.5px] text-muted-foreground border-t border-border/40">
          <span className="capitalize truncate max-w-[120px]">{item.category}</span>
          <div className="flex items-center gap-1 shrink-0">
            {isMarketplace ? (
              <>
                <Download className="h-3 w-3 text-[hsl(var(--success))]" />
                <span>{item.install_count ?? 0}</span>
              </>
            ) : (
              <>
                <Play className="h-3 w-3" />
                <span>{item.use_count ?? 0}</span>
              </>
            )}
          </div>
        </div>
      </Card>
    </motion.div>
  )
}

function RecommendedGrid() {
  const router = useRouter()
  const { data, isLoading } = useRecommendedAssignments(12)

  const items = data?.items ?? []

  const handleClick = useCallback(
    (item: RecommendedItem) => {
      if (item.source === 'marketplace') {
        router.push(`/marketplace?id=${item.id}`)
      } else if (item.type === 'mission') {
        router.push(`/assignments?tab=missions&id=${item.id}`)
      } else {
        router.push(`/assignments?tab=playbooks&id=${item.id}`)
      }
    },
    [router],
  )

  if (!isLoading && items.length === 0) return null

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.25 }}
    >
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Sparkles className="h-4 w-4 text-[hsl(var(--warning))]" />
          <h2 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground">
            Recommended
          </h2>
        </div>
        <button
          onClick={() => router.push('/marketplace')}
          className="text-xs text-muted-foreground hover:text-primary transition-colors flex items-center gap-1"
        >
          Browse Marketplace <span aria-hidden="true">&rarr;</span>
        </button>
      </div>

      {isLoading ? (
        <div className="flex gap-3 overflow-x-auto pb-2 -mx-1 px-1">
          {[...Array(6)].map((_: unknown, i: number) => (
            <Skeleton key={i} className="h-[148px] w-[260px] shrink-0 rounded-lg" />
          ))}
        </div>
      ) : (
        <div
          className="flex gap-3 overflow-x-auto pb-2 -mx-1 px-1 snap-x scroll-smooth"
          style={{ scrollbarWidth: 'thin' }}
        >
          {items.map((item: RecommendedItem, i: number) => (
            <RecommendedCard
              key={`${item.source}-${item.id}`}
              item={item}
              index={i}
              onClick={() => handleClick(item)}
            />
          ))}
        </div>
      )}
    </motion.div>
  )
}

// ── Main Page ──────────────────────────────────────────────────────

const TAB_CONFIG = [
  { value: 'playbooks' as const, label: 'Playbooks', icon: BookOpen },
  { value: 'missions' as const, label: 'Missions', icon: Rocket },
]

export function AssignmentsPage() {
  const searchParams = useSearchParams()
  const router = useRouter()

  const activeTab = resolveTab(searchParams?.get('tab') ?? null)

  const handleTabChange = useCallback(
    (value: string) => {
      router.replace(`/assignments?tab=${value}`)
    },
    [router],
  )

  return (
    <div className="space-y-6">
      {/* ── Page Header ──────────────────────────────────────── */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <div className="flex flex-col gap-1">
          <h1 className="text-2xl md:text-3xl font-bold">
            <ClipboardList className="inline-block h-7 w-7 mr-2 -mt-1 text-primary" />
            Assignments
          </h1>
          <p className="text-sm text-muted-foreground">
            Plan, schedule, and orchestrate work for your crew
          </p>
        </div>
      </motion.div>

      {/* ── Featured Hero Area (US-010) ──────────────────────── */}
      <FeaturedHeroCards />

      {/* ── Contextual Hint (US-011) ─────────────────────────── */}
      <ContextualHeroHint />

      {/* ── Recommended for You (US-016) ─────────────────────── */}
      <RecommendedGrid />

      {/* ── Category Tabs ────────────────────────────────────── */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.2 }}
      >
        <Tabs value={activeTab} onValueChange={handleTabChange} className="space-y-6">
          <TabsList data-tour="assignments-tabs" className="bg-secondary/50">
            {TAB_CONFIG.map((tab) => (
              <TabsTrigger key={tab.value} value={tab.value} className="flex items-center gap-1.5">
                <tab.icon className="h-4 w-4" />
                {tab.label}
              </TabsTrigger>
            ))}
          </TabsList>

          <TabsContent value="playbooks" className="mt-0">
            <AssignmentsPlaybooksGrid />
          </TabsContent>

          <TabsContent value="missions" className="mt-0">
            <AssignmentsMissionsGrid />
          </TabsContent>
        </Tabs>
      </motion.div>
    </div>
  )
}
