'use client'

/**
 * PRD-251 S3.3f (US-308) — publishing a post from its view.
 *
 * - An approved post is scheduled (a date-time picker, in the browser's own
 *   timezone, which the post keeps for showing its slot) or published now, after a
 *   confirmation that names its channels.
 * - A scheduled post shows its slot in its own timezone, and is rescheduled (the
 *   approval stands) or unscheduled (it stays approved).
 * - A missed post says it missed its slot, and is rescheduled or published now.
 *
 * Every call goes through apiClient with POST (use-socials-publish-api.ts); a 409
 * says the post changed and reloads it, as approve does.
 */
import { useState, type FormEvent } from 'react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import type { Workspace } from '@/components/workspace-provider'
import type { SocialPost } from '@/lib/api-client'
import { browserTimezone, slotDateLabel } from '@/lib/social-time'
import {
  usePublishSocialPostNow,
  useScheduleSocialPost,
  useUnscheduleSocialPost,
} from '@/hooks/use-socials-publish-api'
import { canAuthorPosts, channelLabel } from './socials-status'

interface ControlProps {
  post: SocialPost
}

/** The post's channels as the confirmation names them: "LinkedIn, X". */
export function channelNames(post: Pick<SocialPost, 'targets'>): string {
  const names = (post.targets ?? []).map((t) => channelLabel(t.toolkit))
  return Array.from(new Set(names)).join(', ')
}

function ScheduleForm({ post, submit }: ControlProps & { submit: string }) {
  const schedule = useScheduleSocialPost()
  const [wall, setWall] = useState('')
  const tz = browserTimezone()
  const id = `socials-slot-${post.id}`
  const send = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (!wall) return
    // A datetime-local value is the browser's own wall time: that instant, in its timezone.
    schedule.mutate({ postId: post.id, scheduledFor: new Date(wall).toISOString(), timezone: tz })
  }
  return (
    <form onSubmit={send} aria-label={submit} className="flex flex-wrap items-end gap-2">
      <div className="space-y-1">
        <Label htmlFor={id}>
          {submit}: date and time ({tz})
        </Label>
        <Input id={id} type="datetime-local" value={wall} onChange={(e) => setWall(e.target.value)} className="w-auto" />
      </div>
      <Button type="submit" size="sm" disabled={!wall || schedule.isLoading}>
        {submit}
      </Button>
    </form>
  )
}

function PublishNow({ post }: ControlProps) {
  const publish = usePublishSocialPostNow()
  const [confirming, setConfirming] = useState(false)
  if (!confirming) {
    return (
      <Button size="sm" onClick={() => setConfirming(true)} disabled={publish.isLoading}>
        Publish now
      </Button>
    )
  }
  return (
    <div role="alertdialog" aria-label="Publish now" className="space-y-2 rounded-lg border border-border px-3 py-2">
      <p className="text-sm text-foreground">Publish now to {channelNames(post) || 'its channels'}? It goes live at once.</p>
      <div className="flex justify-end gap-2">
        <Button size="sm" variant="ghost" onClick={() => setConfirming(false)}>
          Cancel
        </Button>
        <Button size="sm" onClick={() => (setConfirming(false), publish.mutate({ postId: post.id }))}>
          Publish
        </Button>
      </div>
    </div>
  )
}

function ScheduledControls({ post }: ControlProps) {
  const unschedule = useUnscheduleSocialPost()
  return (
    <>
      <p className="text-sm text-foreground" data-testid="socials-post-slot">
        Scheduled for {slotDateLabel(post.scheduled_for, post.timezone)}
      </p>
      <ScheduleForm post={post} submit="Reschedule" />
      <Button size="sm" variant="outline" onClick={() => unschedule.mutate({ postId: post.id })} disabled={unschedule.isLoading}>
        Unschedule
      </Button>
    </>
  )
}

function MissedControls({ post }: ControlProps) {
  return (
    <>
      <p className="rounded-lg border border-amber-500/40 bg-amber-500/5 px-3 py-2 text-sm text-foreground" role="alert">
        Missed its slot ({slotDateLabel(post.scheduled_for, post.timezone)}). Nothing was published.
      </p>
      <ScheduleForm post={post} submit="Reschedule" />
      <PublishNow post={post} />
    </>
  )
}

export function SocialsPublishControls({ post, role }: ControlProps & { role: Workspace['role'] }) {
  if (!canAuthorPosts(role)) return null
  const body =
    post.status === 'approved' ? (
      <>
        <ScheduleForm post={post} submit="Schedule" />
        <PublishNow post={post} />
      </>
    ) : post.status === 'scheduled' ? (
      <ScheduledControls post={post} />
    ) : post.status === 'missed' ? (
      <MissedControls post={post} />
    ) : null
  if (!body) return null
  return (
    <section aria-label="Publishing" className="space-y-3 rounded-lg border border-border/60 p-3">
      {body}
    </section>
  )
}
