'use client'

import { useCallback } from 'react'
import { useSearchParams, useRouter } from 'next/navigation'
import { motion } from 'framer-motion'
import {
  ClipboardList,
  BookOpen,
  Rocket,
  Sparkles,
} from 'lucide-react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'

// ── Types ──────────────────────────────────────────────────────────

type AssignmentTab = 'playbooks' | 'missions'

const VALID_TABS: ReadonlyArray<AssignmentTab> = ['playbooks', 'missions']

function resolveTab(param: string | null): AssignmentTab {
  if (param && VALID_TABS.includes(param as AssignmentTab)) return param as AssignmentTab
  return 'playbooks'
}

// ── Featured Area Placeholder (US-010) ─────────────────────────────

function FeaturedHeroPlaceholder() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.1 }}
    >
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Hero cards will be added in US-010 */}
        <div className="lg:col-span-2 rounded-lg border border-dashed border-border/50 bg-card/30 p-8 flex items-center justify-center min-h-[160px]">
          <p className="text-sm text-muted-foreground">Hero cards loading in US-010</p>
        </div>
        <div className="rounded-lg border border-dashed border-border/50 bg-card/30 p-8 flex items-center justify-center min-h-[160px]">
          <p className="text-sm text-muted-foreground">Quick actions</p>
        </div>
      </div>
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
      <FeaturedHeroPlaceholder />

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
