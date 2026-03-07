'use client'

import { Rocket, Sparkles } from 'lucide-react'
import { useRouter } from 'next/navigation'
import { motion, useReducedMotion } from 'framer-motion'
import { Button } from '@/components/ui/button'

const CAPABILITIES = [
  'Break complex goals into tasks automatically',
  'Assign the right agents to each task',
  'Track progress with a live dashboard',
  'Produce artifacts (docs, spreadsheets, reports)',
]

export function ActivityMissions() {
  const router = useRouter()
  const prefersReducedMotion = useReducedMotion()

  return (
    <motion.div
      initial={prefersReducedMotion ? false : { opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="glass-card p-8 md:p-12 max-w-2xl mx-auto text-center space-y-6"
    >
      <Rocket className="w-12 h-12 mx-auto text-muted-foreground/30" />

      <div>
        <h2 className="text-2xl font-bold">
          Missions — <span className="gradient-text">Coming Soon</span>
        </h2>
      </div>

      <p className="text-sm text-muted-foreground leading-relaxed max-w-lg mx-auto">
        Missions are big, multi-agent projects that run for hours or days. Give a
        complex brief — like &quot;Prepare the Q1 board deck&quot; — and your AI
        workforce figures out the steps, assigns agents, and delivers results.
      </p>

      <div className="glass-card p-5 text-left space-y-3 max-w-md mx-auto">
        <h3 className="text-sm font-semibold flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-primary" />
          What Missions Can Do
        </h3>
        <ul className="space-y-2">
          {CAPABILITIES.map((item) => (
            <li
              key={item}
              className="flex items-start gap-2 text-sm text-muted-foreground"
            >
              <span className="text-primary mt-0.5">•</span>
              {item}
            </li>
          ))}
        </ul>
      </div>

      <p className="text-xs text-muted-foreground">
        Want early access? Let us know what you&apos;d use it for.
      </p>

      <Button
        variant="outline"
        onClick={() => router.push('/chat')}
        className="border-primary/30 hover:border-primary/50"
      >
        <Rocket className="w-4 h-4 mr-2" />
        Request Early Access
      </Button>
    </motion.div>
  )
}
