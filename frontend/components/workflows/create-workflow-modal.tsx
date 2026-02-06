'use client'

import * as React from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { X, GitBranch, ChevronRight } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { apiClient } from "@/lib/api-client"

export function CreateWorkflowModal({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [step, setStep] = React.useState(1)
  const [name, setName] = React.useState('')
  const [description, setDescription] = React.useState('')
  const [goal, setGoal] = React.useState('')
  const [contextJson, setContextJson] = React.useState('')
  const [showAdvanced, setShowAdvanced] = React.useState(false)
  const [defaultPolicyId, setDefaultPolicyId] = React.useState('')
  const [saving, setSaving] = React.useState(false)
  const [error, setError] = React.useState('')

  async function handleCreate() {
    setSaving(true)
    setError('')
    
    // Parse context JSON if provided
    let contextData = {}
    if (contextJson.trim()) {
      try {
        contextData = JSON.parse(contextJson)
      } catch (e) {
        setError('Invalid JSON in context field')
        setSaving(false)
        return
      }
    }
    
    try {
      const wf = await apiClient.createWorkflow({ 
        name, 
        description,
        goal: goal || undefined,
        context: Object.keys(contextData).length > 0 ? contextData : undefined,
        default_policy_id: defaultPolicyId 
      })
      try { window.dispatchEvent(new Event('workflows:refresh')) } catch {}
      onClose()
      setStep(1)
      setName('')
      setDescription('')
      setGoal('')
      setContextJson('')
      setDefaultPolicyId('')
      setShowAdvanced(false)
    } catch (err: any) {
      setError(err?.message || 'Failed to create workflow')
    } finally { 
      setSaving(false) 
    }
  }

  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onClick={onClose} />
          <motion.div className="fixed inset-0 z-50 flex items-center justify-center p-4" initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.95 }}>
            <Card className="glass-card card-glow w-full max-w-3xl max-h-[90vh] overflow-hidden">
              <CardHeader className="flex flex-row items-center justify-between">
                <CardTitle className="flex items-center space-x-2">
                  <GitBranch className="w-6 h-6" />
                  <span>Create <span className="gradient-text">Workflow</span></span>
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
                      <Label>Name <span className="text-red-400">*</span></Label>
                      <Input value={name} onChange={e=>setName(e.target.value)} placeholder="Workflow name" className="bg-secondary/50" />
                    </div>
                    <div>
                      <Label>Description <span className="text-red-400">*</span></Label>
                      <Textarea rows={3} value={description} onChange={e=>setDescription(e.target.value)} placeholder="Describe what this workflow should accomplish..." className="bg-secondary/50" />
                    </div>
                    
                    {/* Advanced Options Toggle */}
                    <div className="pt-2">
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        onClick={() => setShowAdvanced(!showAdvanced)}
                        className="text-sm text-[hsl(var(--info))] hover:text-[hsl(var(--info))]/80 -ml-2"
                      >
                        <ChevronRight className={`w-4 h-4 mr-1 transition-transform ${showAdvanced ? 'rotate-90' : ''}`} />
                        {showAdvanced ? 'Hide' : 'Show'} Advanced Options
                      </Button>
                    </div>

                    {/* Advanced Options */}
                    {showAdvanced && (
                      <div className="space-y-3 p-3 bg-secondary/20 border border-secondary rounded-lg">
                        <div>
                          <Label htmlFor="goal" className="text-sm">Goal (Optional)</Label>
                          <Input
                            id="goal"
                            value={goal}
                            onChange={(e) => setGoal(e.target.value)}
                            placeholder="e.g., Review PR #123 for security vulnerabilities"
                            className="bg-secondary/50 border-secondary text-sm"
                          />
                          <p className="text-xs text-muted-foreground mt-1">
                            High-level objective - overrides description if provided
                          </p>
                        </div>

                        <div>
                          <Label htmlFor="context" className="text-sm">Context (Optional JSON)</Label>
                          <Textarea
                            id="context"
                            value={contextJson}
                            onChange={(e) => setContextJson(e.target.value)}
                            placeholder={`{\n  "codegraph_project": "my-app",\n  "pr_number": 123\n}`}
                            className="bg-secondary/50 border-secondary font-mono text-xs"
                            rows={4}
                          />
                          <p className="text-xs text-muted-foreground mt-1">
                            Additional context (JSON). Use <code className="bg-secondary/30 px-1 rounded text-xs">codegraph_project</code> to enable code access.
                          </p>
                        </div>
                      </div>
                    )}
                  </TabsContent>
                  <TabsContent value="step-2" className="space-y-4">
                    <div>
                      <Label>Default Policy ID</Label>
                      <Input value={defaultPolicyId} onChange={e=>setDefaultPolicyId(e.target.value)} placeholder="code_assistant" className="bg-secondary/50" />
                    </div>
                  </TabsContent>
                </Tabs>
                {/* Error Display */}
                {error && (
                  <div className="p-3 bg-[hsl(var(--destructive))]/10 border border-[hsl(var(--destructive))]/50 rounded-lg text-sm text-[hsl(var(--destructive))]">
                    {error}
                  </div>
                )}

                <div className="flex justify-between items-center mt-8 pt-6 border-t border-border/30">
                  <Button variant="outline" onClick={()=>setStep(Math.max(1, step-1))} disabled={step===1}>Previous</Button>
                  <div className="text-sm text-muted-foreground">Step {step} of 2</div>
                  {step<2 ? (
                    <Button variant="outline" onClick={()=>setStep(2)} disabled={!name || !description}>Next</Button>
                  ) : (
                    <Button onClick={handleCreate} disabled={!name || saving} aria-busy={saving}>
                      {saving ? 'Creating...' : 'Create Workflow'}
                    </Button>
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


