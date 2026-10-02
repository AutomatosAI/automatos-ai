'use client'

/**
 * PRD-251B US-B109 — the editor's draft: started from the post (or empty for a new one),
 * started again when another post opens or the server answers a save, and `dirty` while
 * what is on screen differs from what the post holds.
 */
import { useEffect, useMemo, useState } from 'react'

import type { SocialPost } from '@/lib/api-client'
import { draftFromPost, editorTargets, newDraft, postFields, slotInput, type EditorDraft } from './editor-model'

function snapshot(draft: EditorDraft): string {
  return JSON.stringify([postFields(draft, []), editorTargets(draft), slotInput(draft.slot), draft.footageOn])
}

export function useEditorDraft(post: SocialPost | null) {
  const [draft, setDraft] = useState<EditorDraft>(() => (post ? draftFromPost(post) : newDraft()))
  const version = post ? `${post.id}:${post.updated_at}` : 'new'

  // The server's answer (a save, an upload, another tab) is the new starting point.
  useEffect(() => {
    if (post) setDraft(draftFromPost(post))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [version])

  const baseline = useMemo(() => (post ? snapshot(draftFromPost(post)) : snapshot(newDraft())), [post])
  return { draft, setDraft, dirty: snapshot(draft) !== baseline }
}
