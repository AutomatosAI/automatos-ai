'use client'

/**
 * PRD-251B US-B109 — the editor's draft: started from the post (or empty for a new one),
 * started again when another post opens or the server answers a save, and `dirty` while
 * what is on screen differs from what the post holds. A change on the server to the
 * post's footage alone (its AI options being made, US-B305) is taken into the draft
 * without starting it again, so nothing typed meanwhile is lost.
 */
import { useEffect, useMemo, useRef, useState } from 'react'

import type { SocialPost } from '@/lib/api-client'
import { draftFromPost, editorTargets, newDraft, postFields, slotInput, type EditorDraft } from './editor-model'

function snapshot(draft: EditorDraft): string {
  return JSON.stringify([postFields(draft, []), editorTargets(draft), slotInput(draft.slot), draft.footageOn, draft.footage])
}

function withoutFootage(draft: EditorDraft): string {
  return snapshot({ ...draft, footage: {} })
}

export function useEditorDraft(post: SocialPost | null) {
  const [draft, setDraft] = useState<EditorDraft>(() => (post ? draftFromPost(post) : newDraft()))
  const version = post ? `${post.id}:${post.updated_at}` : 'new'
  const shown = useRef<SocialPost | null>(post)

  // The server's answer (a save, an upload, another tab) is the new starting point; a
  // change to the footage alone is taken in as it is.
  useEffect(() => {
    if (!post) return
    const before = shown.current
    shown.current = post
    const next = draftFromPost(post)
    const footageOnly = !!before && before.id === post.id && withoutFootage(draftFromPost(before)) === withoutFootage(next)
    setDraft((current) => (footageOnly ? { ...current, footage: next.footage } : next))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [version])

  const baseline = useMemo(() => (post ? snapshot(draftFromPost(post)) : snapshot(newDraft())), [post])
  return { draft, setDraft, dirty: snapshot(draft) !== baseline }
}
