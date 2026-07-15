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
  const isMobile = typeof window !== 'undefined' && window.innerWidth <= 640

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
    markTourSkipped('welcome', userId)
    onOpenChange(false)

    // Mobile: welcome modal only, no Shepherd tour
    if (isMobile) return

    setIsStarting(true)
    // Small delay for modal close animation, then launch the Chat page
    // Shepherd tour (the main tour we built — no separate welcome tour).
    setTimeout(async () => {
      const { createChatTour } = await import('@/lib/shepherd/tours/chat-tour')
      const tour = createChatTour(userId)
      tour.start()
      setIsStarting(false)
    }, 300)
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent size="md">
        <DialogHeader>
          <div className="flex items-center gap-3 mb-1">
            <Image
              src="/brand/automatos-mark.png"
              alt="Automatos AI"
              width={40}
              height={40}
              className="rounded-lg"
            />
            <div>
              <DialogTitle className="text-xl">
                Welcome to <span className="text-white">Automatos</span>{' '}
                <span className="gradient-text">AI</span>
              </DialogTitle>
              <DialogDescription className="text-muted-foreground text-sm">
                Your intelligent automation platform
              </DialogDescription>
            </div>
          </div>
        </DialogHeader>

        <div className="space-y-4 py-2">
          {/* Friendly overview */}
          <p className="text-sm text-foreground/80 leading-relaxed">
            Quick 60-second orientation — we&apos;ll show you where everything lives.
          </p>

          <div className="grid grid-cols-2 gap-2">
            <div className="p-3 rounded-lg bg-secondary/50 border border-border">
              <div className="font-medium text-sm text-foreground/80 mb-0.5">
                AI Agents
              </div>
              <div className="text-xs text-muted-foreground">
                Autonomous workers for email, data, research &amp; more
              </div>
            </div>
            <div className="p-3 rounded-lg bg-secondary/50 border border-border">
              <div className="font-medium text-sm text-foreground/80 mb-0.5">
                1,000+ Integrations
              </div>
              <div className="text-xs text-muted-foreground">
                Gmail, Slack, Jira, GitHub and all your tools
              </div>
            </div>
            <div className="p-3 rounded-lg bg-secondary/50 border border-border">
              <div className="font-medium text-sm text-foreground/80 mb-0.5">
                Playbooks
              </div>
              <div className="text-xs text-muted-foreground">
                Multi-agent workflows with no-code builder
              </div>
            </div>
            <div className="p-3 rounded-lg bg-secondary/50 border border-border">
              <div className="font-medium text-sm text-foreground/80 mb-0.5">
                AI Chat
              </div>
              <div className="text-xs text-muted-foreground">
                Get help anytime from Auto, your assistant
              </div>
            </div>
          </div>

          {/* Business Intake CTA — PRD-130 / PRD-203 O·S1 */}
          <div className="p-4 rounded-lg bg-primary/10 border border-primary/30" data-testid="business-intake-cta">
            <div className="flex items-start gap-3">
              <div className="mt-0.5">
                <Building2 className="w-5 h-5 text-primary" />
              </div>
              <div className="flex-1">
                <div className="font-medium text-foreground/90 mb-1">
                  Tell Automatos about your business
                </div>
                <div className="text-sm text-muted-foreground">
                  Share your domain and Auto will scan your site, build a Knowledge Graph,
                  and draft a Mission Zero plan in under 3 minutes.
                </div>
                <Button
                  size="sm"
                  onClick={handleStartIntake}
                  className="mt-3 bg-primary hover:bg-primary/90 text-primary-foreground"
                  data-testid="business-intake-start"
                >
                  Start Business Intake
                  <Building2 className="w-4 h-4 ml-2" />
                </Button>
              </div>
            </div>
          </div>

          {/* Tour CTA — desktop only */}
          {!isMobile && (
            <div
              className="p-3 rounded-lg bg-primary/10 border border-primary/30 cursor-pointer hover:bg-primary/20 transition-colors"
              onClick={handleStartTour}
            >
              <div className="flex items-center gap-3">
                <Sparkles className="w-5 h-5 text-primary flex-shrink-0" />
                <div className="flex-1">
                  <div className="font-medium text-sm text-foreground/90">
                    Take a quick tour
                  </div>
                  <div className="text-xs text-muted-foreground">
                    Step by step — press <kbd className="px-1 py-0.5 text-xs bg-secondary rounded">ESC</kbd> to exit anytime
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Actions */}
          <div className="flex items-center justify-between">
            <Button
              variant="ghost"
              size="sm"
              onClick={handleSkip}
              className="text-muted-foreground hover:text-foreground text-sm"
            >
              {isMobile ? 'Close' : "Skip, I'll explore on my own"}
            </Button>
            <Button
              onClick={handleStartTour}
              disabled={isStarting}
              variant="outline"
              size="sm"
            >
              {isMobile ? "Let's Go" : isStarting ? 'Starting...' : 'Start Tour'}
              <Sparkles className="w-4 h-4 ml-2" />
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}
