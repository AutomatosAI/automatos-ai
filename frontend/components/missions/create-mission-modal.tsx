'use client'

import { useState } from 'react'
import { Target, Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { Label } from '@/components/ui/label'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from '@/components/ui/dialog'
import { useCreateMission } from '@/hooks/use-missions-api'
import { useMissionStore } from '@/stores/mission-store'
import { toast } from 'sonner'
import { useRouter } from 'next/navigation'

interface CreateMissionModalProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

export function CreateMissionModal({ open, onOpenChange }: CreateMissionModalProps) {
  const router = useRouter()
  const createMission = useCreateMission()
  const setActivePlanningMissionId = useMissionStore((s) => s.setActivePlanningMissionId)

  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [tags, setTags] = useState('')

  const isSubmitting = createMission.isLoading

  const handleSubmit = () => {
    const goalParts: string[] = []
    if (name.trim()) goalParts.push(name.trim())
    if (description.trim()) goalParts.push(description.trim())

    const goal = goalParts.join(': ')
    if (!goal) {
      toast.error('Please enter a mission name or description')
      return
    }

    // Build config from optional fields
    const config: Record<string, unknown> = {}
    const tagList = tags
      .split(',')
      .map((t) => t.trim())
      .filter(Boolean)
    if (tagList.length > 0) config.tags = tagList
    if (name.trim()) config.name = name.trim()

    createMission.mutate(
      { goal, ...(Object.keys(config).length > 0 ? { config } : {}) },
      {
        onSuccess: (mission) => {
          setActivePlanningMissionId(mission.id)
          toast.success('Mission created — plan is being generated')
          onOpenChange(false)
          resetForm()
          router.push(`/missions/${mission.id}` as any)
        },
        onError: (err) => {
          toast.error(err.message || 'Failed to create mission')
        },
      },
    )
  }

  const resetForm = () => {
    setName('')
    setDescription('')
    setTags('')
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Target className="w-5 h-5 text-primary" />
            New Mission
          </DialogTitle>
          <DialogDescription>
            Define a goal for your AI workforce. The system will decompose it into tasks,
            assign agents, and generate a plan for your approval.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-2">
          <div className="space-y-2">
            <Label htmlFor="mission-name">Mission Name</Label>
            <Input
              id="mission-name"
              placeholder="e.g. Research top AI agent frameworks"
              value={name}
              onChange={(e) => setName(e.target.value)}
              autoFocus
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="mission-description">Description</Label>
            <Textarea
              id="mission-description"
              placeholder="Describe what you want to accomplish, any constraints, output format..."
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              rows={4}
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="mission-tags">Tags</Label>
            <Input
              id="mission-tags"
              placeholder="e.g. research, competitive-analysis, urgent"
              value={tags}
              onChange={(e) => setTags(e.target.value)}
            />
            <p className="text-xs text-muted-foreground">Comma-separated</p>
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)} disabled={isSubmitting}>
            Cancel
          </Button>
          <Button onClick={handleSubmit} disabled={isSubmitting || (!name.trim() && !description.trim())}>
            {isSubmitting ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Creating...
              </>
            ) : (
              <>
                <Target className="w-4 h-4 mr-2" />
                Create Mission
              </>
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
