'use client'

/**
 * PRD-251D (E4, US-D104) — a new post starts with its channels ticked (`startingKinds` in
 * editor-model.ts): once, when the connected channels and the workspace's posts have loaded,
 * and only while the person has ticked nothing. A tick or a story switch the person changes
 * first is theirs and is never overridden; a saved post opens with its own channels.
 */
import { useEffect, useRef, type Dispatch, type SetStateAction } from 'react'

import type { SocialChannel, SocialPost } from '@/lib/api-client'
import { useSocialPosts } from '@/hooks/use-socials-api'
import { withStartingChannels, type EditorDraft } from './editor-model'

/** Applies the starting channels to a new post; returns what to call when the person changes a
 * channel themselves, so the starting ones never land over their choice. */
export function useStartingChannels(
  post: SocialPost | null,
  channels: ReadonlyArray<SocialChannel> | undefined,
  setDraft: Dispatch<SetStateAction<EditorDraft>>,
): () => void {
  const { data } = useSocialPosts()
  const posts = data?.posts
  const settled = useRef(post !== null)

  useEffect(() => {
    if (settled.current || !channels || !posts) return
    settled.current = true
    setDraft((draft) => (Object.keys(draft.kinds).length > 0 ? draft : withStartingChannels(draft, channels, posts)))
  }, [channels, posts, setDraft])

  return () => {
    settled.current = true
  }
}
