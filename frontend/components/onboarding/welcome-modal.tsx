'use client'

import { useState } from 'react'
import Image from 'next/image'
import { useRouter } from 'next/navigation'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Sparkles, Building2 } from 'lucide-react'
import { markTourSkipped } from '@/lib/shepherd/tour-storage'

interface WelcomeModalProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  userId: string
}

export function WelcomeModal({ open, onOpenChange, userId }: WelcomeModalProps) {
  const [isStarting, setIsStarting] = useState(false)
  const router = useRouter()

  const handleSkip = () => {
    markTourSkipped('welcome', userId)
    onOpenChange(false)
  }

  const handleStartIntake = () => {
    markTourSkipped('welcome', userId)
    onOpenChange(false)
    setTimeout(() => router.push('/onboarding/wizard'), 200)
  }

  const handleStartTour = () => {
    setIsStarting(true)
    onOpenChange(false)

    // Small delay for modal close animation
    setTimeout(async () => {
      // Dynamic import — shepherd.js accesses `window` at module init,
      // so it must NOT be imported at the top level (breaks SSR).
      const { createWelcomeTour } = await import('@/lib/shepherd/tours/welcome-tour')
      const tour = createWelcomeTour(userId)
      tour.start()
      setIsStarting(false)
    }, 300)
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <div className="flex items-center gap-3 mb-2">
            <Image
              src="/brand/automatos-mark.png"
              alt="Automatos AI"
              width={48}
              height={48}
              className="rounded-lg"
            />
            <div>
              <DialogTitle className="text-2xl">
                Welcome to <span className="text-white">Automatos</span>{' '}
                <span className="gradient-text">AI</span>
              </DialogTitle>
              <DialogDescription className="text-gray-400 mt-1">
                Your intelligent automation platform
              </DialogDescription>
            </div>
          </div>
        </DialogHeader>

        <div className="space-y-6 py-4">
          {/* Friendly overview */}
          <div>
            <p className="text-gray-300 leading-relaxed">
              Quick 60-second orientation — we&apos;ll show you where everything lives.
              Each page also has its own mini-tour that appears on your first visit.
            </p>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="p-4 rounded-lg bg-gray-800/50 border border-gray-700">
              <div className="font-medium text-sm text-gray-300 mb-1">
                AI Agents
              </div>
              <div className="text-xs text-gray-400">
                Create autonomous workers for email, data, research, and more
              </div>
            </div>
            <div className="p-4 rounded-lg bg-gray-800/50 border border-gray-700">
              <div className="font-medium text-sm text-gray-300 mb-1">
                150+ Integrations
              </div>
              <div className="text-xs text-gray-400">
                Connect to Gmail, Slack, Jira, GitHub, and all your tools
              </div>
            </div>
            <div className="p-4 rounded-lg bg-gray-800/50 border border-gray-700">
              <div className="font-medium text-sm text-gray-300 mb-1">
                Workflows &amp; Recipes
              </div>
              <div className="text-xs text-gray-400">
                Build complex automations with no-code visual builder
              </div>
            </div>
            <div className="p-4 rounded-lg bg-gray-800/50 border border-gray-700">
              <div className="font-medium text-sm text-gray-300 mb-1">
                AI Chat
              </div>
              <div className="text-xs text-gray-400">
                Get help anytime from your AI assistant
              </div>
            </div>
          </div>

          {/* Business Intake CTA — PRD-130 */}
          <div className="p-4 rounded-lg bg-primary/10 border border-primary/30">
            <div className="flex items-start gap-3">
              <div className="mt-0.5">
                <Building2 className="w-5 h-5 text-primary" />
              </div>
              <div className="flex-1">
                <div className="font-medium text-gray-200 mb-1">
                  Tell Automatos about your business
                </div>
                <div className="text-sm text-gray-400">
                  Share your domain and we&apos;ll scan your site, build a knowledge graph,
                  and draft a Mission Zero plan in under 3 minutes.
                </div>
                <Button
                  size="sm"
                  onClick={handleStartIntake}
                  className="mt-3 bg-primary hover:bg-primary/90 text-primary-foreground"
                >
                  Start Business Intake
                  <Building2 className="w-4 h-4 ml-2" />
                </Button>
              </div>
            </div>
          </div>

          {/* Tour CTA */}
          <div className="p-4 rounded-lg bg-orange-500/10 border border-orange-500/30">
            <div className="flex items-start gap-3">
              <div className="mt-0.5">
                <Sparkles className="w-5 h-5 text-orange-400" />
              </div>
              <div className="flex-1">
                <div className="font-medium text-gray-200 mb-1">
                  Or take a quick tour
                </div>
                <div className="text-sm text-gray-400">
                  We&apos;ll guide you step by step — you can skip or exit anytime by pressing{' '}
                  <kbd className="px-1.5 py-0.5 text-xs bg-gray-700 rounded">ESC</kbd>
                </div>
              </div>
            </div>
          </div>

          {/* Actions */}
          <div className="flex items-center justify-between pt-2">
            <Button
              variant="ghost"
              onClick={handleSkip}
              className="text-gray-400 hover:text-gray-200"
            >
              Skip, I&apos;ll explore on my own
            </Button>
            <Button
              onClick={handleStartTour}
              disabled={isStarting}
              variant="outline"
            >
              {isStarting ? 'Starting...' : 'Start Tour'}
              <Sparkles className="w-4 h-4 ml-2" />
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}
