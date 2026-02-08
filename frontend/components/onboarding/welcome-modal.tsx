'use client'

import { useState } from 'react'
import Image from 'next/image'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Sparkles } from 'lucide-react'
import { createFirstLoginTour } from '@/lib/shepherd/first-login-tour'
import { markOnboardingSkipped } from '@/lib/shepherd/tour-storage'

interface WelcomeModalProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  userId: string
}

export function WelcomeModal({ open, onOpenChange, userId }: WelcomeModalProps) {
  const [isStarting, setIsStarting] = useState(false)

  const handleSkip = () => {
    markOnboardingSkipped(userId)
    onOpenChange(false)
  }

  const handleStartTour = () => {
    setIsStarting(true)
    onOpenChange(false)

    // Small delay for modal close animation
    setTimeout(() => {
      const tour = createFirstLoginTour(userId)
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
              We&apos;ll walk you through setting up your workspace, designing your first AI agent,
              choosing a persona and model, connecting tools, and exploring recipes and workflows.
              It only takes a couple of minutes!
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

          {/* Tour CTA */}
          <div className="p-4 rounded-lg bg-orange-500/10 border border-orange-500/30">
            <div className="flex items-start gap-3">
              <div className="mt-0.5">
                <Sparkles className="w-5 h-5 text-orange-400" />
              </div>
              <div className="flex-1">
                <div className="font-medium text-gray-200 mb-1">
                  Take a 2-minute tour
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
              className="bg-gradient-to-r from-orange-500 to-orange-600 hover:from-orange-600 hover:to-orange-700 text-white"
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
