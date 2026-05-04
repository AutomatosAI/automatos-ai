'use client'

import * as React from 'react'
import { Play } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from '@/components/ui/dialog'
import { apiClient } from "@/lib/api-client"

export function RunWorkflowModal({ open, onClose, id }: { open: boolean; onClose: () => void; id: string }) {
  const [input, setInput] = React.useState('{"query":"hello"}')
  const [running, setRunning] = React.useState(false)

  async function runOnce() {
    setRunning(true)
    try {
      const payload = input ? JSON.parse(input) : {}
      await apiClient.runWorkflow(id, payload)
      try { window.dispatchEvent(new Event('workflows:refresh')) } catch {}
      onClose()
    } catch { /* ignore */ } finally { setRunning(false) }
  }

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent size="md">
        <DialogHeader>
          <DialogTitle className="flex items-center space-x-2">
            <Play className="w-5 h-5" />
            <span>Run <span className="gradient-text">Workflow</span></span>
          </DialogTitle>
        </DialogHeader>
        <Textarea rows={10} className="font-mono text-xs" value={input} onChange={e=>setInput(e.target.value)} />
        <DialogFooter>
          <Button onClick={runOnce} aria-busy={running}>Run</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
