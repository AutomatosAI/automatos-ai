'use client'

/**
 * PRD-251 S3.3f (US-308) — publishing a post from its view.
 *
 * - An approved post is scheduled (a date-time picker, in the browser's own
 *   timezone, which the post keeps for showing its slot) or published now.
 * - A scheduled post shows its slot in its own timezone, and is rescheduled (the
 *   approval stands), unscheduled (it stays approved) or published now.
 * - A missed post says it missed its slot, and is rescheduled or published now.
 * - One Publish now (17 Nov 2026, Gerard: "are you sure, are you really sure?"): the button
 *   names the channels, "Publish now to Instagram, X", and publishes on its one click. There is
 *   no second step.
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
import { canAuthorPosts, channelNames } from './socials-status'

export const PUBLISH_NOW = 'Publish now'
/** What a post with no channel is published to: the server refuses it (409, no channels). */
export const ITS_CHANNELS = 'its channels'

interface ControlProps {
  post: SocialPost
}

/** "Publish now to Instagram, X": the one button that publishes the post at once. */
export function publishNowLabel(post: Pick<SocialPost, 'targets'>): string {
  return `${PUBLISH_NOW} to ${channelNames(post) || ITS_CHANNELS}`
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
  return (
    <Button size="sm" onClick={() => publish.mutate({ postId: post.id })} disabled={publish.isLoading}>
      {publishNowLabel(post)}
    </Button>
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
      <PublishNow post={post} />
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
