'use client'

/**
 * PRD-130: Business Intake Wizard Shell
 * ======================================
 *
 * Modal shell mirroring create-agent-modal.tsx — same Card/glass style,
 * same Tabs-based stepper, same close + animation pattern.
 *
 * Phase 1 = 7 steps, ending at the Mission Zero Draft Plan review.
 * Mission 1 (team provisioning) is parked as Phase 2.
 *
 * Step 5 is now driven by Server-Sent Events — the scrape endpoint
 * returns 202 immediately, the background pipeline emits progress to
 * Redis, and `useWizardProgress` streams those events into the terminal
 * feed. When the pipeline emits `stage=complete` we pull the fresh
 * profile and advance to Step 6.
 */

import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import { motion, AnimatePresence } from 'framer-motion'
import { X, Sparkles } from 'lucide-react'
import { toast } from 'react-hot-toast'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'

import {
  useStartWizard,
  useScanDomain,
  useScrapeSelected,
  usePatchProfile,
  useGenerateDraftPlan,
  useFetchProfile,
  type ScanResponse,
  type BusinessProfilePayload,
} from '@/hooks/use-wizard-api'
import { useWizardProgress } from '@/hooks/use-wizard-progress'

import { Step1Goals } from './step-1-goals'
import { Step2Domain } from './step-2-domain'
import { Step3Scanning } from './step-3-scanning'
import { Step4PageChecklist } from './step-4-page-checklist'
import { Step5Intake } from './step-5-intake'
import { Step6ProfileEditor } from './step-6-profile-editor'

interface WizardShellProps {
  open: boolean
  onClose: () => void
  onComplete?: () => void
}

interface WizardState {
  goals: string[]
  domain: string
  profileId: string | null
  scan: ScanResponse | null
  selectedUrls: string[]
  profile: BusinessProfilePayload | null
}

const INITIAL: WizardState = {
  goals: [],
  domain: '',
  profileId: null,
  scan: null,
  selectedUrls: [],
  profile: null,
}

export function WizardShell({ open, onClose, onComplete }: WizardShellProps) {
  const router = useRouter()
  const [step, setStep] = useState(1)
  const [state, setState] = useState<WizardState>(INITIAL)
  const [progressActive, setProgressActive] = useState(false)

  const startMutation = useStartWizard()
  const scanMutation = useScanDomain()
  const scrapeMutation = useScrapeSelected()
  const patchMutation = usePatchProfile()
  const planMutation = useGenerateDraftPlan()
  const fetchProfileMutation = useFetchProfile()

  const progress = useWizardProgress({
    profileId: state.profileId,
    active: progressActive,
  })

  const reset = () => {
    setState(INITIAL)
    setStep(1)
    setProgressActive(false)
    progress.reset()
  }

  const handleClose = () => {
    setProgressActive(false)
    onClose()
    setTimeout(reset, 200)
  }

  // ---- Step transitions --------------------------------------------------

  const handleStartScan = async () => {
    try {
      const start = await startMutation.mutateAsync({
        domain: state.domain,
        goals: state.goals,
      })
      setState(s => ({ ...s, profileId: start.profile_id }))
      setStep(3)
      const scan = await scanMutation.mutateAsync(start.profile_id)
      setState(s => ({
        ...s,
        scan,
        selectedUrls: scan.must_have_urls.slice(),
      }))
      setStep(4)
    } catch (err: any) {
      toast.error(`Scan failed: ${err?.message || 'unknown error'}`)
    }
  }

  const handleStartScrape = async () => {
    if (!state.profileId) return
    // Advance to step 5 and open the SSE stream BEFORE we fire the
    // scrape POST so we don't miss the first events.
    setStep(5)
    progress.reset()
    setProgressActive(true)
    try {
      await scrapeMutation.mutateAsync({
        profileId: state.profileId,
        selectedUrls: state.selectedUrls,
      })
      // 202 accepted — the background pipeline is running. Events will
      // stream in via the progress hook; completion is handled in the
      // effect below.
    } catch (err: any) {
      // Only the initial POST can land here — the actual pipeline
      // failure comes through SSE. Stay on step 5 and show the error
      // inline so the user doesn't lose their page selection.
      toast.error(`Could not start intake: ${err?.message || 'unknown error'}`)
      setProgressActive(false)
    }
  }

  // ---- SSE completion handler --------------------------------------------

  useEffect(() => {
    if (!progressActive) return
    if (progress.state !== 'complete') return
    if (!state.profileId) return

    let cancelled = false
    ;(async () => {
      try {
        const profile = await fetchProfileMutation.mutateAsync(state.profileId!)
        if (cancelled) return
        setState(s => ({ ...s, profile }))
        setProgressActive(false)
        setStep(6)
      } catch (err: any) {
        toast.error(
          `Could not load profile after intake: ${err?.message || 'unknown error'}`
        )
      }
    })()

    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [progress.state, progressActive, state.profileId])

  useEffect(() => {
    if (progress.state === 'failed') {
      toast.error('Intake pipeline failed — see the log for details')
    }
  }, [progress.state])

  const handleSaveProfile = async (patch: Partial<BusinessProfilePayload>) => {
    if (!state.profileId) return
    try {
      await patchMutation.mutateAsync({ profileId: state.profileId, patch })
      setState(s => ({
        ...s,
        profile: s.profile ? { ...s.profile, ...patch } : (patch as BusinessProfilePayload),
      }))
      toast.success('Profile saved')
    } catch (err: any) {
      toast.error(`Save failed: ${err?.message || 'unknown error'}`)
    }
  }

  const handleGeneratePlan = async () => {
    if (!state.profileId) return
    try {
      const plan = await planMutation.mutateAsync(state.profileId)
      toast.success('Mission Zero launched — review the plan and hit Approve.')
      router.push(`/missions/${plan.mission_id}`)
      onComplete?.()
      handleClose()
    } catch (err: any) {
      toast.error(`Mission launch failed: ${err?.message || 'unknown error'}`)
    }
  }

  return (
    <AnimatePresence>
      {open && (
        <>
          {/* Backdrop */}
          <motion.div
            className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={handleClose}
          />

          {/* Modal */}
          <motion.div
            className="fixed inset-0 z-50 flex items-center justify-center p-4"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ duration: 0.2 }}
          >
            <Card className="bg-card border-border w-full max-w-4xl max-h-[90vh] overflow-hidden">
              <CardHeader className="flex flex-row items-center justify-between border-b border-border/30">
                <CardTitle className="flex items-center space-x-3">
                  <Sparkles className="w-6 h-6 text-primary" />
                  <div>
                    <span className="text-xl">
                      Business <span className="gradient-text">Intake Wizard</span>
                    </span>
                    <p className="text-sm text-muted-foreground font-normal">
                      Tell Automatos about your business in under 3 minutes
                    </p>
                  </div>
                </CardTitle>
                <Button variant="ghost" size="icon" onClick={handleClose}>
                  <X className="w-5 h-5" />
                </Button>
              </CardHeader>

              <CardContent className="overflow-y-auto p-6">
                <Tabs value={`step-${step}`} className="space-y-6">
                  <TabsList className="w-full justify-start gap-1 bg-secondary/50 flex-wrap h-auto">
                    <TabsTrigger value="step-1" disabled={step < 1}>1. Goals</TabsTrigger>
                    <TabsTrigger value="step-2" disabled={step < 2}>2. Domain</TabsTrigger>
                    <TabsTrigger value="step-3" disabled={step < 3}>3. Scan</TabsTrigger>
                    <TabsTrigger value="step-4" disabled={step < 4}>4. Pages</TabsTrigger>
                    <TabsTrigger value="step-5" disabled={step < 5}>5. Intake</TabsTrigger>
                    <TabsTrigger value="step-6" disabled={step < 6}>6. Profile</TabsTrigger>
                  </TabsList>

                  <TabsContent value="step-1" className="space-y-4 max-h-[55vh] overflow-y-auto">
                    <Step1Goals
                      selected={state.goals}
                      onChange={(goals) => setState(s => ({ ...s, goals }))}
                      onNext={() => setStep(2)}
                    />
                  </TabsContent>

                  <TabsContent value="step-2" className="space-y-4 max-h-[55vh] overflow-y-auto">
                    <Step2Domain
                      domain={state.domain}
                      onChange={(domain) => setState(s => ({ ...s, domain }))}
                      onBack={() => setStep(1)}
                      onScan={handleStartScan}
                      isScanning={startMutation.isLoading || scanMutation.isLoading}
                    />
                  </TabsContent>

                  <TabsContent value="step-3" className="space-y-4 max-h-[55vh] overflow-y-auto">
                    <Step3Scanning domain={state.domain} />
                  </TabsContent>

                  <TabsContent value="step-4" className="space-y-4 max-h-[55vh] overflow-y-auto">
                    {state.scan && (
                      <Step4PageChecklist
                        scan={state.scan}
                        selected={state.selectedUrls}
                        onChange={(urls) => setState(s => ({ ...s, selectedUrls: urls }))}
                        onBack={() => setStep(3)}
                        onIngest={handleStartScrape}
                      />
                    )}
                  </TabsContent>

                  <TabsContent value="step-5" className="space-y-4 max-h-[55vh] overflow-y-auto">
                    <Step5Intake
                      pageCount={state.selectedUrls.length}
                      events={progress.events}
                      state={progress.state}
                    />
                  </TabsContent>

                  <TabsContent value="step-6" className="space-y-4 max-h-[55vh] overflow-y-auto">
                    {state.profile && (
                      <Step6ProfileEditor
                        profile={state.profile}
                        scrape={null}
                        onSave={handleSaveProfile}
                        isSaving={patchMutation.isLoading}
                        onBack={() => setStep(4)}
                        onGeneratePlan={handleGeneratePlan}
                        isGenerating={planMutation.isLoading}
                      />
                    )}
                  </TabsContent>

                </Tabs>
              </CardContent>
            </Card>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  )
}
