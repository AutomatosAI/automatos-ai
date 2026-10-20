'use client'

import { useState } from 'react'
import { Sparkles, Pencil, Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { toast } from 'sonner'
import { useCreateBlogMission } from '@/hooks/use-blogs-api'

export type BlogCreateMode = 'manual' | 'mission'

const DEFAULT_MISSION_CATEGORY = 'AI & Automation'

function modeButtonClass(active: boolean): string {
  return `flex items-center gap-2 px-4 py-1.5 rounded-md text-sm font-medium transition ${
    active ? 'bg-background shadow-sm' : 'text-muted-foreground hover:text-foreground'
  }`
}

/** New Post only: write it yourself, or have agents write it. */
export function BlogModeToggle({ mode, onChange }: { mode: BlogCreateMode; onChange: (mode: BlogCreateMode) => void }) {
  return (
    <div className="mt-4 flex gap-2 p-1 bg-muted/40 rounded-lg w-fit">
      <button type="button" onClick={() => onChange('manual')} className={modeButtonClass(mode === 'manual')}>
        <Pencil className="w-4 h-4" /> Write Manually
      </button>
      <button type="button" onClick={() => onChange('mission')} className={modeButtonClass(mode === 'mission')}>
        <Sparkles className="w-4 h-4" /> Have Agents Write It
      </button>
    </div>
  )
}

/** Mission mode: a topic and a category; agents research, write and queue the draft. */
export function BlogMissionForm({ onClose }: { onClose: () => void }) {
  const [missionTopic, setMissionTopic] = useState('')
  const [missionCategory, setMissionCategory] = useState(DEFAULT_MISSION_CATEGORY)
  const missionMutation = useCreateBlogMission()
  const isMissionStarting = missionMutation.isLoading

  const handleStartMission = async () => {
    const topic = missionTopic.trim()
    if (!topic) {
      toast.error('Topic is required')
      return
    }
    await missionMutation.mutateAsync({
      topic,
      category: missionCategory.trim() || undefined,
    })
    onClose()
  }

  return (
    <div className="mt-6 space-y-4 max-w-xl">
      <div className="text-sm text-muted-foreground">
        Pick a topic. Agents will research, write, edit, generate a cover image,
        and queue the draft for your review. Takes 5-10 minutes.
      </div>

      <div className="space-y-2">
        <Label htmlFor="mission-topic">Topic</Label>
        <Input
          id="mission-topic"
          placeholder="e.g. Multi-agent orchestration for Shopify stores"
          value={missionTopic}
          onChange={(e) => setMissionTopic(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && missionTopic.trim()) {
              e.preventDefault()
              handleStartMission()
            }
          }}
          autoFocus
        />
        <p className="text-xs text-muted-foreground">
          Be specific. The more concrete the topic, the better the post.
        </p>
      </div>

      <div className="space-y-2">
        <Label htmlFor="mission-category">Category</Label>
        <Input
          id="mission-category"
          placeholder={DEFAULT_MISSION_CATEGORY}
          value={missionCategory}
          onChange={(e) => setMissionCategory(e.target.value)}
        />
      </div>

      <div className="flex items-center justify-end gap-2 pt-2">
        <Button variant="ghost" onClick={onClose} disabled={isMissionStarting}>
          Cancel
        </Button>
        <Button onClick={handleStartMission} disabled={!missionTopic.trim() || isMissionStarting}>
          {isMissionStarting ? (
            <>
              <Loader2 className="w-4 h-4 mr-1 animate-spin" /> Starting…
            </>
          ) : (
            <>
              <Sparkles className="w-4 h-4 mr-1" /> Start Mission
            </>
          )}
        </Button>
      </div>
    </div>
  )
}
