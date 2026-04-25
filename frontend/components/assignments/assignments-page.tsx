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
  Store,
} from 'lucide-react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Card, CardContent, CardFooter } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { CreateMissionModal } from '@/components/missions/create-mission-modal'
import { CreatePlaybookModal } from '@/components/workflows/create-playbook-modal'
import { CreateTaskDialog } from '@/components/activity/board/create-task-dialog'
import { AssignmentsPlaybooksGrid } from '@/components/assignments/assignments-playbooks-grid'
import { AssignmentsMissionsGrid } from '@/components/assignments/assignments-missions-grid'
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
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.1 }}
    >
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Mission — hero card (large, top-left) */}
        <Card className="lg:col-span-2 overflow-hidden border-primary/20 bg-gradient-to-br from-primary/10 via-card/80 to-card/50 backdrop-blur-sm hover:border-primary/40 transition-colors">
          <CardContent className="p-6 flex flex-col justify-between min-h-[200px]">
            <div>
              <div className="flex items-center gap-2 mb-2">
                <Rocket className="h-6 w-6 text-primary" />
                <h3 className="text-xl font-bold">Mission</h3>
              </div>
              <p className="text-sm text-muted-foreground leading-relaxed">
                Big complex work. 6&ndash;9 agents, field memory, parallel reasoning.
              </p>
            </div>
            <div className="mt-4">
              <Button onClick={() => setMissionOpen(true)}>
                Start <span aria-hidden="true">&rarr;</span>
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Right column: Playbook + Plan + Task stacked */}
        <div className="flex flex-col gap-4">
          {/* Playbook card (medium) */}
          <Card className="border-border/50 bg-card/50 backdrop-blur-sm hover:border-primary/20 transition-colors">
            <CardContent className="p-4">
              <div className="flex items-center gap-2 mb-1.5">
                <BookOpen className="h-5 w-5 text-primary" />
                <h3 className="font-semibold">Playbook</h3>
              </div>
              <p className="text-xs text-muted-foreground mb-3">
                Routines. Multi-task, schedulable, triggerable, reusable.
              </p>
              <Button size="sm" variant="secondary" onClick={() => setPlaybookOpen(true)}>
                + Playbook
              </Button>
            </CardContent>
          </Card>

          {/* Plan card (medium) */}
          <Card className="border-border/50 bg-card/50 backdrop-blur-sm hover:border-primary/20 transition-colors">
            <CardContent className="p-4">
              <div className="flex items-center gap-2 mb-1.5">
                <Lightbulb className="h-5 w-5 text-[hsl(var(--warning))]" />
                <h3 className="font-semibold">Plan</h3>
              </div>
              <p className="text-xs text-muted-foreground mb-3">
                Iterate with Auto first.
              </p>
              <Button
                size="sm"
                variant="secondary"
                onClick={() => router.push('/chat?mode=plan&from=assignments')}
              >
                + Plan
              </Button>
            </CardContent>
          </Card>

          {/* Task card (small) */}
          <Card className="border-border/50 bg-card/50 backdrop-blur-sm hover:border-primary/20 transition-colors">
            <CardContent className="p-3 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Zap className="h-4 w-4 text-muted-foreground" />
                <div>
                  <h3 className="text-sm font-semibold">Task</h3>
                  <p className="text-xs text-muted-foreground">Quick single action.</p>
                </div>
              </div>
              <Button size="sm" variant="ghost" onClick={() => setTaskOpen(true)}>
                + Task
              </Button>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Create modals */}
      <CreateMissionModal open={missionOpen} onOpenChange={setMissionOpen} />
      <CreatePlaybookModal open={playbookOpen} onClose={() => setPlaybookOpen(false)} />
      <CreateTaskDialog open={taskOpen} onOpenChange={setTaskOpen} />
    </motion.div>
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

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: index * 0.05 }}
    >
      <Card
        role="button"
        tabIndex={0}
        className="cursor-pointer glass-card card-glow hover:border-primary/20 transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-primary/50 h-full flex flex-col"
        onClick={onClick}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault()
            onClick()
          }
        }}
      >
        <CardContent className="p-4 flex-1">
          <div className="flex items-start gap-3 mb-2">
            <div className="w-10 h-10 rounded-xl bg-primary/20 border border-primary/30 flex items-center justify-center shrink-0">
              <Icon className="h-5 w-5 text-primary" />
            </div>
            <div className="flex-1 min-w-0">
              <h3 className="font-semibold text-sm line-clamp-1">{item.name}</h3>
              <p className="text-xs text-muted-foreground line-clamp-2 mt-0.5">
                {item.description}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2 mt-3">
            <Badge variant="outline" className="text-xs border-border text-muted-foreground">
              {item.category}
            </Badge>
            {isMarketplace && (
              <Badge className="text-xs bg-primary/20 text-primary border-primary/30">
                <Store className="h-3 w-3 mr-1" />
                Marketplace
              </Badge>
            )}
          </div>
        </CardContent>
        <CardFooter className="pt-2 pb-3 px-4 border-t border-border flex items-center justify-between text-xs text-muted-foreground">
          {isMarketplace ? (
            <div className="flex items-center gap-1">
              <Download className="h-3.5 w-3.5 text-[hsl(var(--success))]" />
              <span>{item.install_count ?? 0} installs</span>
            </div>
          ) : (
            <div className="flex items-center gap-1">
              <Play className="h-3.5 w-3.5 text-muted-foreground" />
              <span>{item.use_count ?? 0} runs</span>
            </div>
          )}
          <span className="capitalize">{item.type}</span>
        </CardFooter>
      </Card>
    </motion.div>
  )
}

function RecommendedGrid() {
  const router = useRouter()
  const { data, isLoading } = useRecommendedAssignments(8)

  const items = data?.items ?? []

  const handleClick = useCallback(
    (item: RecommendedItem) => {
      if (item.source === 'marketplace') {
        router.push(`/marketplace?id=${item.id}`)
      } else if (item.type === 'mission') {
        router.push(`/command-center?tab=missions&id=${item.id}`)
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
      <div className="flex items-center gap-2 mb-3">
        <Sparkles className="h-5 w-5 text-[hsl(var(--warning))]" />
        <h2 className="text-lg font-semibold">Recommended for You</h2>
      </div>

      {isLoading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {[...Array(4)].map((_: unknown, i: number) => (
            <Skeleton key={i} className="h-48 rounded-lg" />
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
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

      {/* ── Recommended for You (US-016) ─────────────────────── */}
      <RecommendedGrid />
    </div>
  )
}
