'use client'

/**
 * PRD-130: Business Intake Wizard route
 * =====================================
 *
 * Hosts the WizardShell modal. Defaults to open; closing returns the user
 * to the dashboard. Supports `?force=1` for dev re-runs (no extra logic
 * needed — the route is dev-friendly because the wizard always opens here).
 */

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { WizardShell } from '@/components/wizard/wizard-shell'

export default function WizardPage() {
  const router = useRouter()
  const [open, setOpen] = useState(true)

  const [completed, setCompleted] = useState(false)

  const handleClose = () => {
    setOpen(false)
    // Only redirect to activity if the wizard wasn't completed
    // (completed = mission launched, shell handles its own redirect)
    if (!completed) {
      setTimeout(() => router.push('/assignments?tab=missions'), 250)
    }
  }

  return (
    <div className="min-h-screen bg-background">
      <WizardShell
        open={open}
        onClose={handleClose}
        onComplete={() => setCompleted(true)}
      />
    </div>
  )
}
