'use client'

import { useState } from 'react'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Sparkles } from 'lucide-react'
import { createFirstLoginTour } from '@/lib/shepherd/first-login-tour'
import { markOnboardingSkipped } from '@/lib/shepherd/tour-storage'

interface WelcomeModalProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

export function WelcomeModal({ open, onOpenChange }: WelcomeModalProps) {
  const [isStarting, setIsStarting] = useState(false)

  const handleSkip = () => {
    markOnboardingSkipped()
    onOpenChange(false)
  }

  const handleStartTour = () => {
    setIsStarting(true)
    onOpenChange(false)

    // Small delay for modal close animation
    setTimeout(() => {
      const tour = createFirstLoginTour()
      tour.start()
      setIsStarting(false)
    }, 300)
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <div className="flex items-center gap-3 mb-2">
            <div className="p-3 rounded-full bg-gradient-to-br from-blue-500 to-purple-600">
              <Sparkles className="w-6 h-6 text-white" />
            </div>
            <div>
              <DialogTitle className="text-2xl">Welcome to Automatos AI</DialogTitle>
              <DialogDescription className="text-gray-400 mt-1">
                Your intelligent automation platform
              </DialogDescription>
            </div>
          </div>
        </DialogHeader>

        <div className="space-y-6 py-4">
          {/* Quick Overview */}
          <div>
            <h3 className="font-semibold text-gray-200 mb-3">
              What you can do with Automatos:
            </h3>
            <div className="grid grid-cols-2 gap-3">
              <div className="p-4 rounded-lg bg-gray-800/50 border border-gray-700">
                <div className="font-medium text-sm text-gray-300 mb-1">
                  🤖 AI Agents
                </div>
                <div className="text-xs text-gray-400">
                  Create autonomous workers for email, data, research, and more
                </div>
              </div>
              <div className="p-4 rounded-lg bg-gray-800/50 border border-gray-700">
                <div className="font-medium text-sm text-gray-300 mb-1">
                  🔌 150+ Integrations
                </div>
                <div className="text-xs text-gray-400">
                  Connect to Gmail, Slack, Jira, GitHub, and all your tools
                </div>
              </div>
              <div className="p-4 rounded-lg bg-gray-800/50 border border-gray-700">
                <div className="font-medium text-sm text-gray-300 mb-1">
                  ⚡ Workflows
                </div>
                <div className="text-xs text-gray-400">
                  Build complex automations with no-code visual builder
                </div>
              </div>
              <div className="p-4 rounded-lg bg-gray-800/50 border border-gray-700">
                <div className="font-medium text-sm text-gray-300 mb-1">
                  💬 AI Chat
                </div>
                <div className="text-xs text-gray-400">
                  Get help anytime from your AI assistant
                </div>
              </div>
            </div>
          </div>

          {/* Tour CTA */}
          <div className="p-4 rounded-lg bg-blue-500/10 border border-blue-500/30">
            <div className="flex items-start gap-3">
              <div className="mt-0.5">
                <Sparkles className="w-5 h-5 text-blue-400" />
              </div>
              <div className="flex-1">
                <div className="font-medium text-gray-200 mb-1">
                  Take a 2-minute tour
                </div>
                <div className="text-sm text-gray-400">
                  We'll walk you through creating your first email assistant agent.
                  You can skip or exit anytime by pressing <kbd className="px-1.5 py-0.5 text-xs bg-gray-700 rounded">ESC</kbd>
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
              Skip, I'll explore on my own
            </Button>
            <Button
              onClick={handleStartTour}
              disabled={isStarting}
              className="bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700"
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
