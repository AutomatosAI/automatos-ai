'use client'

/**
 * PRD-251B US-B109 — the post editor (Editor.dc.html), at ?post=new (the post is created on
 * the first Save draft) or ?post=<id>. The header holds Save draft, Render preview and
 * Submit for approval; the cards, in order: Brief (Redraft with Auto), Format, Channels and
 * sizes, Look (Template · Upload · Library · AI-made), Claims and sources, When; beside them the
 * copy and the preview. Save draft writes the post's fields (POST or PATCH), its channels
 * (PUT /targets) and its slot (PUT /slot, only when it moved); the server checks it all.
 * F254: a new post is one post: once a Save, Render or Submit has created it, every later
 * try edits it, and a try that failed after creating it opens it.
 */
import { useMemo, useRef } from 'react'

import type { Workspace } from '@/components/workspace-provider'
import type { SocialPost } from '@/lib/api-client'
import { useSocialChannels } from '@/hooks/use-socials-composer'
import {
  useMakeAiOptions, usePickAiOption, useRedraft, usePickLibraryMedia, useRenderEditorPreview, useSaveEditor, useSocialFootageSources,
  useSocialTemplates, useSubmitEditor, useUploadEditorMedia, type EditorSave,
} from '@/hooks/use-socials-editor'
import { channelsOverLimit } from '../socials-composer-model'
import { EditorBriefCard } from './editor-brief-card'
import { EditorChannelsCard } from './editor-channels-card'
import { EditorClaimsCard } from './editor-claims-card'
import { EditorFormatCard } from './editor-format-card'
import { EditorLookCard } from './editor-look-card'
import { EditorPreviewColumn } from './editor-preview-column'
import { EditorWhenCard } from './editor-when-card'
import {
  editorTargets, postFields, slidesOf, slotChanged, slotInput, videoSlotsOf, withChannelTicked, withFormat, withProposal, withSlides,
} from './editor-model'
import { SocialsEditorActivity } from './socials-editor-activity'
import { SocialsEditorHeader } from './socials-editor-header'
import type { GoTo } from './studio-route'
import { useEditorDraft } from './use-editor-draft'
import { Hint } from './editor-ui'

/** A new post's four steps, in one line under its header. */
export const NEW_POST_STEPS = 'Write a brief (or let Auto redraft it), choose the look, tick the channels, then Render preview and Submit for approval.'

interface SocialsEditorProps {
  role: Workspace['role']
  /** The post, or null for a new one. */
  post: SocialPost | null
  go: GoTo
}

function useEditorCalls() {
  return {
    save: useSaveEditor(), render: useRenderEditorPreview(), submit: useSubmitEditor(),
    upload: useUploadEditorMedia(), pick: usePickLibraryMedia(), redraft: useRedraft(),
    aiMake: useMakeAiOptions(), aiPick: usePickAiOption(),
  }
}

export function SocialsEditor({ role, post, go }: SocialsEditorProps) {
  const { draft, setDraft } = useEditorDraft(post)
  const { data: channelData, isLoading: channelsLoading } = useSocialChannels()
  const channels = useMemo(() => channelData ?? [], [channelData])
  const templates = useSocialTemplates(draft.format)
  const footage = useSocialFootageSources(draft.format !== 'text')
  const chosen = templates.data?.find((t) => t.id === draft.templateId) ?? null
  const imageSlots = chosen?.image_slots ?? []
  const calls = useEditorCalls()
  // F254: the post a Save, Render or Submit created, kept even when a later step fails, so
  // the next try edits it instead of creating another.
  const created = useRef<string | null>(null)

  const payload = (): EditorSave => ({
    postId: post?.id ?? created.current,
    fields: postFields(draft, chosen?.footage_slots ?? [], imageSlots),
    targets: editorTargets(draft),
    slot: slotChanged(post, draft.slot) ? slotInput(draft.slot) : undefined,
    onCreated: (saved) => { created.current = saved.id },
  })
  const opened = {
    onSuccess: (saved: SocialPost) => (post ? undefined : go({ post: saved.id })),
    // F254: a try that failed after creating the post opens that post, as it now is.
    onError: () => (post || !created.current ? undefined : go({ post: created.current })),
  }
  const busy = calls.save.isLoading ? 'save' : calls.render.isLoading ? 'render' : calls.submit.isLoading ? 'submit' : null
  const pickTemplate = (templateId: string | null) => {
    const lengths = templates.data?.find((t) => t.id === templateId)?.durations ?? []
    setDraft((d) => ({ ...d, templateId, lengthSeconds: d.format === 'video' ? lengths.find((l) => l === d.lengthSeconds) ?? lengths[0] ?? null : null }))
  }
  const redraft = () => {
    const ticked = Object.keys(draft.kinds)
    calls.redraft.mutate(
      { brief: draft.brief, channels: ticked.length ? ticked : undefined, format: draft.format, template_id: draft.templateId, length_seconds: draft.lengthSeconds },
      { onSuccess: (proposal) => setDraft((d) => withProposal(d, proposal)) },
    )
  }
  const mediaBusy = calls.upload.isLoading || calls.pick.isLoading

  return (
    <div className="socials-editor flex flex-col gap-5">
      <SocialsEditorHeader
        post={post} draft={draft} busy={busy} overLimit={channelsOverLimit(draft, channels).length > 0}
        onTitle={(title) => setDraft((d) => ({ ...d, title }))}
        onBack={() => go({ view: 'calendar', post: null })}
        onSave={() => calls.save.mutate(payload(), opened)}
        onRender={() => calls.render.mutate({ ...payload(), video: draft.format === 'video' }, opened)}
        onSubmit={() => calls.submit.mutate(payload(), opened)}
      />
      {!post && <Hint>{NEW_POST_STEPS}</Hint>}
      <div className="grid items-start gap-5 lg:grid-cols-[minmax(0,0.95fr)_minmax(0,1.05fr)]">
        <div className="flex min-w-0 flex-col gap-4">
          <EditorBriefCard brief={draft.brief} busy={calls.redraft.isLoading} canRedraft onChange={(brief) => setDraft((d) => ({ ...d, brief }))} onRedraft={redraft} />
          <EditorFormatCard
            draft={draft} durations={chosen?.durations ?? []} footageSlots={videoSlotsOf(chosen?.footage_slots ?? [], imageSlots)}
            footage={footage.data?.kinds.video} post={post} canEdit slides={slidesOf(draft)}
            onChange={setDraft} onFormat={(format) => setDraft((d) => withFormat(d, format, channels))}
            onSlides={(n) => setDraft((d) => withSlides(d, n))}
          />
          <EditorChannelsCard
            channels={channels} loading={channelsLoading} draft={draft} templateSizes={chosen?.sizes ?? []}
            onTick={(channel, on) => setDraft((d) => withChannelTicked(d, channel, on))}
            onOptions={(toolkit, options) => setDraft((d) => ({ ...d, options: { ...d.options, [toolkit]: options } }))}
          />
          {draft.format !== 'text' && (
            <EditorLookCard
              templates={templates.data ?? []} loading={templates.isLoading} chosen={draft.templateId} onPick={pickTemplate} busy={mediaBusy}
              onUpload={(file) => calls.upload.mutate({ ...payload(), file }, opened)}
              onLibrary={(item) => calls.pick.mutate({ ...payload(), item }, opened)}
              ai={{
                post, imageSlots, images: footage.data?.kinds.image, aiBusy: calls.aiMake.isLoading || calls.aiPick.isLoading,
                onAiMake: (imageSlot, prompt) => calls.aiMake.mutate({ ...payload(), imageSlot, prompt }, opened),
                onAiPick: (slot, name) => post && calls.aiPick.mutate({ postId: post.id, slot, name }),
              }}
            />
          )}
          <EditorClaimsCard
            schema={chosen?.variables_schema ?? null} variables={draft.variables} sources={draft.sources} examples={chosen?.sample_data}
            onChange={(variables, sources) => setDraft((d) => ({ ...d, variables, sources }))}
          />
          <EditorWhenCard slot={draft.slot} onChange={(slot) => setDraft((d) => ({ ...d, slot }))} />
          {post && <SocialsEditorActivity post={post} role={role} />}
        </div>
        <EditorPreviewColumn
          post={post} draft={draft} channels={channels} templateSizes={chosen?.sizes ?? []}
          onBase={(base) => setDraft((d) => ({ ...d, base }))}
          onChannelCopy={(toolkit, text) => setDraft((d) => ({ ...d, perChannel: { ...d.perChannel, [toolkit]: text } }))}
        />
      </div>
    </div>
  )
}
