'use client'

import * as React from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { X, Pencil } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { apiClient } from '@/lib/api'

export function EditWorkflowModal({ open, onClose, id }: { open: boolean; onClose: () => void; id: string }) {
  const [name, setName] = React.useState('')
  const [description, setDescription] = React.useState('')
  const [defaultPolicyId, setDefaultPolicyId] = React.useState('')
  const [owner, setOwner] = React.useState('')
  const [tags, setTags] = React.useState<string[]>([])
  const [tagInput, setTagInput] = React.useState('')
  const [saving, setSaving] = React.useState(false)

  React.useEffect(() => {
    let alive = true
    if (!open) return
    apiClient.getWorkflow(id).then(wf => {
      if (!alive || !wf) return
      setName(wf.name || '')
      setDescription(wf.description || '')
      setDefaultPolicyId(wf.default_policy_id || '')
      setOwner(wf.owner || '')
      setTags(Array.isArray(wf.tags) ? wf.tags : [])
    })
    return () => { alive = false }
  }, [open, id])

  async function save() {
    setSaving(true)
    try {
      await apiClient.updateWorkflow(id, { name, description, default_policy_id: defaultPolicyId, owner, tags })
      try { window.dispatchEvent(new Event('workflows:refresh')) } catch {}
      onClose()
    } finally { setSaving(false) }
  }

  function addTagFromInput() {
    const raw = tagInput.trim()
    if (!raw) return
    const valid = /^[-_a-zA-Z0-9]{1,20}$/.test(raw)
    if (!valid) return
    if (!tags.includes(raw)) setTags(prev => [...prev, raw])
    setTagInput('')
  }

  function removeTag(t: string) {
    setTags(prev => prev.filter(x => x !== t))
  }

  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onClick={onClose} />
          <motion.div className="fixed inset-0 z-50 flex items-center justify-center p-4" initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.95 }}>
            <Card className="glass-card w-full max-w-3xl max-h-[85vh] overflow-hidden">
              <CardHeader className="flex flex-row items-center justify-between">
                <CardTitle className="flex items-center space-x-2"><Pencil className="w-5 h-5" /><span>Edit Workflow</span></CardTitle>
                <Button variant="ghost" size="icon" onClick={onClose}><X className="w-5 h-5" /></Button>
              </CardHeader>
              <CardContent className="grid gap-4">
                <div>
                  <Label>Name</Label>
                  <Input value={name} onChange={e=>setName(e.target.value)} placeholder="Workflow name" className="bg-secondary/50" />
                </div>
                <div>
                  <Label>Description</Label>
                  <Textarea rows={3} value={description} onChange={e=>setDescription(e.target.value)} className="bg-secondary/50" />
                </div>
                <div className="grid gap-2">
                  <Label>Owner</Label>
                  <Input value={owner} onChange={e=>setOwner(e.target.value)} placeholder="team@company or username" className="bg-secondary/50" />
                  <div className="text-xs text-muted-foreground">Free-form for now. We can wire this to users later.</div>
                </div>
                <div className="grid gap-2">
                  <Label>Tags</Label>
                  <div className="flex items-center gap-2">
                    <Input value={tagInput} onChange={e=>setTagInput(e.target.value)} onKeyDown={e=>{ if (e.key==='Enter'){ e.preventDefault(); addTagFromInput(); } }} placeholder="Add tag and press Enter" className="bg-secondary/50" />
                    <Button type="button" variant="outline" onClick={addTagFromInput}>Add</Button>
                  </div>
                  <div className="flex flex-wrap gap-2 pt-1">
                    {tags.map(t => (
                      <span key={t} className="px-2 py-0.5 rounded-full border bg-secondary/40 text-xs flex items-center gap-1">
                        {t}
                        <button className="opacity-70 hover:opacity-100" onClick={()=>removeTag(t)} aria-label={`remove ${t}`}>×</button>
                      </span>
                    ))}
                    {!tags.length && <span className="text-xs text-muted-foreground">No tags</span>}
                  </div>
                  <div className="text-xs text-muted-foreground">Allowed: letters, numbers, "-" and "_" (max 20 chars)</div>
                </div>
                <div>
                  <Label>Default Policy ID</Label>
                  <Input value={defaultPolicyId} onChange={e=>setDefaultPolicyId(e.target.value)} placeholder="code_assistant" className="bg-secondary/50" />
                </div>
                <div className="flex justify-end gap-2 pt-2">
                  <Button variant="outline" onClick={onClose}>Cancel</Button>
                  <Button onClick={save} aria-busy={saving} className="gradient-accent">Save</Button>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  )
}


