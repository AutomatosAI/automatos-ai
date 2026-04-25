'use client'

import { useCallback, useState } from 'react'
import { useSearchParams, useRouter } from 'next/navigation'
import { motion } from 'framer-motion'
import {
  ClipboardList,
  BookOpen,
  Rocket,
  Sparkles,
  Lightbulb,
  Zap,
} from 'lucide-react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Card, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { CreateMissionModal } from '@/components/missions/create-mission-modal'
import { CreatePlaybookModal } from '@/components/workflows/create-playbook-modal'
import { CreateTaskDialog } from '@/components/activity/board/create-task-dialog'

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

// ── Hero Hint Placeholder (US-011) ─────────────────────────────────

function HeroHintPlaceholder() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.15 }}
    >
      <div className="rounded-lg border border-dashed border-border/50 bg-card/30 p-4">
        <p className="text-sm text-muted-foreground">Contextual hint — wired in US-011</p>
      </div>
    </motion.div>
  )
}

// ── Recommended Placeholder (US-016) ───────────────────────────────

function RecommendedPlaceholder() {
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
      <div className="rounded-lg border border-dashed border-border/50 bg-card/30 p-6 flex items-center justify-center">
        <p className="text-sm text-muted-foreground">Suggestions wired in US-016</p>
      </div>
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
      <HeroHintPlaceholder />

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
            {/* Playbooks grid wired in US-012 */}
            <div className="rounded-lg border border-dashed border-border/50 bg-card/30 p-8 flex items-center justify-center min-h-[200px]">
              <p className="text-sm text-muted-foreground">Playbooks grid — US-012</p>
            </div>
          </TabsContent>

          <TabsContent value="missions" className="mt-0">
            {/* Missions grid wired in US-013 */}
            <div className="rounded-lg border border-dashed border-border/50 bg-card/30 p-8 flex items-center justify-center min-h-[200px]">
              <p className="text-sm text-muted-foreground">Missions grid — US-013</p>
            </div>
          </TabsContent>
        </Tabs>
      </motion.div>

      {/* ── Recommended for You (US-016) ─────────────────────── */}
      <RecommendedPlaceholder />
    </div>
  )
}
