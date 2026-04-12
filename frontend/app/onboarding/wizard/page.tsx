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

  const handleClose = () => {
    setOpen(false)
    setTimeout(() => router.push('/dashboard'), 250)
  }

  return (
    <div className="min-h-screen bg-background">
      <WizardShell
        open={open}
        onClose={handleClose}
        onComplete={() => router.push('/chat')}
      />
    </div>
  )
}
