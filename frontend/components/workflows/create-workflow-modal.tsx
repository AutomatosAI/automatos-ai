'use client'

import * as React from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { X, GitBranch } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { apiClient } from '@/lib/api'

export function CreateWorkflowModal({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [step, setStep] = React.useState(1)
  const [name, setName] = React.useState('')
  const [description, setDescription] = React.useState('')
  const [defaultPolicyId, setDefaultPolicyId] = React.useState('')
  const [saving, setSaving] = React.useState(false)

  async function handleCreate() {
    setSaving(true)
    try {
      const wf = await apiClient.createWorkflow({ name, description, default_policy_id: defaultPolicyId })
      try { window.dispatchEvent(new Event('workflows:refresh')) } catch {}
      onClose()
      setStep(1); setName(''); setDescription(''); setDefaultPolicyId('')
    } finally { setSaving(false) }
  }

  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onClick={onClose} />
          <motion.div className="fixed inset-0 z-50 flex items-center justify-center p-4" initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.95 }}>
            <Card className="glass-card w-full max-w-3xl max-h-[90vh] overflow-hidden">
              <CardHeader className="flex flex-row items-center justify-between">
                <CardTitle className="flex items-center space-x-2">
                  <GitBranch className="w-6 h-6" />
                  <span>Create Workflow</span>
                </CardTitle>
                <Button variant="ghost" size="icon" onClick={onClose}><X className="w-5 h-5" /></Button>
              </CardHeader>
              <CardContent className="overflow-y-auto">
                <Tabs value={`step-${step}`} className="space-y-6">
                  <TabsList className="grid w-full grid-cols-2 bg-secondary/50">
                    <TabsTrigger value="step-1" disabled={step < 1}>1. Details</TabsTrigger>
                    <TabsTrigger value="step-2" disabled={step < 2}>2. Policy</TabsTrigger>
                  </TabsList>
                  <TabsContent value="step-1" className="space-y-4">
                    <div>
                      <Label>Name</Label>
                      <Input value={name} onChange={e=>setName(e.target.value)} placeholder="Workflow name" className="bg-secondary/50" />
                    </div>
                    <div>
                      <Label>Description</Label>
                      <Textarea rows={3} value={description} onChange={e=>setDescription(e.target.value)} className="bg-secondary/50" />
                    </div>
                  </TabsContent>
                  <TabsContent value="step-2" className="space-y-4">
                    <div>
                      <Label>Default Policy ID</Label>
                      <Input value={defaultPolicyId} onChange={e=>setDefaultPolicyId(e.target.value)} placeholder="code_assistant" className="bg-secondary/50" />
                    </div>
                  </TabsContent>
                </Tabs>
                <div className="flex justify-between items-center mt-8 pt-6 border-t border-border/30">
                  <Button variant="outline" onClick={()=>setStep(Math.max(1, step-1))} disabled={step===1}>Previous</Button>
                  <div className="text-sm text-muted-foreground">Step {step} of 2</div>
                  {step<2 ? (
                    <Button onClick={()=>setStep(2)} disabled={!name || !description} className="gradient-accent">Next</Button>
                  ) : (
                    <Button onClick={handleCreate} disabled={!name} className="gradient-accent" aria-busy={saving}>Create Workflow</Button>
                  )}
                </div>
              </CardContent>
            </Card>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  )
}


