'use client'

import * as React from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { X, Play } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { apiClient } from '@/lib/api'

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
    <AnimatePresence>
      {open && (
        <>
          <motion.div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onClick={onClose} />
          <motion.div className="fixed inset-0 z-50 flex items-center justify-center p-4" initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.95 }}>
            <Card className="glass-card w-full max-w-xl max-h-[80vh] overflow-hidden">
              <CardHeader className="flex flex-row items-center justify-between">
                <CardTitle className="flex items-center space-x-2"><Play className="w-5 h-5" /><span>Run Workflow</span></CardTitle>
                <Button variant="ghost" size="icon" onClick={onClose}><X className="w-5 h-5" /></Button>
              </CardHeader>
              <CardContent className="grid gap-3">
                <Textarea rows={10} className="font-mono text-xs" value={input} onChange={e=>setInput(e.target.value)} />
                <div className="flex justify-end"><Button onClick={runOnce} aria-busy={running} className="gradient-accent">Run</Button></div>
              </CardContent>
            </Card>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  )
}


