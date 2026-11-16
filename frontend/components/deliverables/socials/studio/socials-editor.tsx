'use client'

/**
 * PRD-251B US-B109 — the post editor (Editor.dc.html), at ?post=new (the post is created on
 * the first Save draft) or ?post=<id>. The header holds Save draft, Render preview and
 * Submit for approval; the cards, in order: Brief (Write with Auto), Format, Channels and
 * sizes, Look (Template · Upload · Library · AI-made), Claims and sources, When; beside them the
 * copy and the preview. PRD-264 EUAI-08a: a post with a visual has the AI-generated label card
 * (EditorAiLabelCard), on by default while it carries AI media. Save draft writes the post's
 * fields (POST or PATCH), its channels (PUT /targets) and its slot (PUT /slot, only when it moved); the server checks it all.
 * F254: a new post is one post: once a Save, Render or Submit has created it, every later
 * try edits it, and a try that failed after creating it opens it.
 * PRD-251D Wave 1: a post that waits for approval is approved here, with the Queue's controls
 * (US-D101); Submit for approval makes the file with the full render (US-D102, editor-submit.ts);
 * a new post starts with its channels ticked (US-D104); a channel not connected yet is connected
 * from the Channels card (US-D105).
 */
import { useEffect, useMemo, useRef, useState } from 'react'

import type { Workspace } from '@/components/workspace-provider'
import type { SocialPost } from '@/lib/api-client'
import { useSocialChannels } from '@/hooks/use-socials-composer'
import {
  useMakeAiOptions, usePickAiOption, useRedraft, usePickLibraryMedia, useRenderEditorPreview, useSaveEditor, useSocialFootageSources,
  useSocialTemplates, useSubmitEditor, useUploadEditorMedia, type EditorSave,
} from '@/hooks/use-socials-editor'
import { autoNeeds, channelsOverLimit } from '../socials-composer-model'
import { schemaAtLength } from '../socials-variables-form'
import { EditorAiLabelCard } from './editor-ai-label-card'
import { EditorApproveCard } from './editor-approve-card'
import { EditorBriefCard } from './editor-brief-card'
import { EditorChannelsCard } from './editor-channels-card'
import { EditorClaimsCard } from './editor-claims-card'
import { EditorFormatCard } from './editor-format-card'
import { EditorLookCard } from './editor-look-card'
import { photoSpotsOf } from './editor-look-photo'
import { EditorPreviewColumn } from './editor-preview-column'
import { saveOutdatesFile, submitNote, submitRenders, type SubmitInput } from './editor-submit'
import { EditorWhenCard } from './editor-when-card'
import {
  editorTargets, givenFields, isOwnFilePost, postFields, slidesOf, slotChanged, slotInput, videoSlotsOf, withChannelTicked, withFormat,
  withProposal, withSlides, withStory,
} from './editor-model'
import { SocialsEditorActivity } from './socials-editor-activity'
import { DeletePostButton } from './delete-post-button'
import { SocialsEditorHeader } from './socials-editor-header'
import type { GoTo } from './studio-route'
import { useEditorDraft } from './use-editor-draft'
import { useStartingChannels } from './use-starting-channels'
import { Hint } from './editor-ui'

/** A new post's four steps, in one line under its header. */
export const NEW_POST_STEPS = 'Write a brief, choose the look and check the channels. Auto writes the caption and the words on the image: fill a field in yourself only when you want your own words. Then Submit for approval makes the post and sends it to be approved; Render preview first if you want a look.'
/** A post that is the person's own file (an upload or a Library pick): its next steps. */
export const OWN_FILE_STEPS = 'Your file is the post, so there is nothing to render: tick the channels, check each one in the preview, then Submit for approval.'
/** PRD-251C (US-C303): a picture of your own is cropped to each channel's shape by a render. */
export const OWN_STILL_STEPS = "Your picture is the post: tick the channels. Render preview crops it to each one's shape (no render minutes) so you can check each; Submit for approval crops it too and sends it to be approved."

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
  const { draft, setDraft, dirty } = useEditorDraft(post)
  const { data: channelData, isLoading: channelsLoading, refetch: refetchChannels } = useSocialChannels()
  const channels = useMemo(() => channelData ?? [], [channelData])
  const touchChannels = useStartingChannels(post, channelData, setDraft)  // PRD-251D US-D104
  const templates = useSocialTemplates(draft.format)
  const footage = useSocialFootageSources(draft.format !== 'text')
  const chosen = templates.data?.find((t) => t.id === draft.templateId) ?? null
  const imageSlots = chosen?.image_slots ?? []
  const photoSpots = useMemo(() => photoSpotsOf(chosen), [chosen])
  const calls = useEditorCalls()
  const [needs, setNeeds] = useState<string | null>(null)  // F378: what the last redraft asked of the owner
  // A change saved here to what the file shows leaves the file older than the words until a render
  // makes it again: Approve waits and Submit is offered (editor-submit.ts `saveOutdatesFile`).
  const [fileOutdated, setFileOutdated] = useState(false)
  useEffect(() => {
    if (post?.status === 'rendering') setFileOutdated(false)
  }, [post?.status])
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
    const given = givenFields(draft)  // what the person wrote stays; Auto writes the rest
    calls.redraft.mutate(
      {
        brief: draft.brief, channels: ticked.length ? ticked : undefined, format: draft.format, template_id: draft.templateId,
        length_seconds: draft.lengthSeconds, ...(given ? { given_variables: given } : {}),
      },
      {
        onSuccess: (proposal) => {
          setDraft((d) => withProposal(d, proposal))
          setNeeds(autoNeeds(proposal.questions))
        },
      },
    )
  }
  const saveDraft = () => {
    const outdates = saveOutdatesFile(post, draft)
    calls.save.mutate(payload(), {
      ...opened,
      onSuccess: (saved) => {
        if (outdates) setFileOutdated(true)
        return opened.onSuccess(saved)
      },
    })
  }
  const mediaBusy = calls.upload.isLoading || calls.pick.isLoading
  const ownFile = isOwnFilePost(post, draft)
  const ownStill = ownFile && draft.format === 'image'  // PRD-251C US-C303: rendered into a crop per channel
  // PRD-251D (E1): what Submit for approval makes first, decided from what is on screen.
  const submit: SubmitInput = { format: draft.format, ownFile, lengthSeconds: draft.lengthSeconds, sizes: (chosen?.sizes ?? []).length }

  return (
    <div className="socials-editor flex flex-col gap-5">
      <SocialsEditorHeader
        post={post} draft={draft} busy={busy} overLimit={channelsOverLimit(draft, channels).length > 0} nothingToRender={ownFile && !ownStill}
        submitNote={submitNote(submit)} fileOutdated={fileOutdated}
        onTitle={(title) => setDraft((d) => ({ ...d, title }))}
        onBack={() => go({ view: 'calendar', post: null })}
        onSave={() => saveDraft()}
        onRender={() => calls.render.mutate({ ...payload(), video: draft.format === 'video' }, opened)}
        onSubmit={() => calls.submit.mutate({ ...payload(), render: submitRenders(submit) }, opened)}
        extra={post && <DeletePostButton post={post} role={role} onDeleted={() => go({ view: 'calendar', post: null })} />}
      />
      {!post && <Hint>{NEW_POST_STEPS}</Hint>}
      {ownFile && <Hint>{ownStill ? OWN_STILL_STEPS : OWN_FILE_STEPS}</Hint>}
      {post && <EditorApproveCard post={post} role={role} dirty={dirty} fileOutdated={fileOutdated} />}
      <div className="grid items-start gap-5 lg:grid-cols-[minmax(0,0.95fr)_minmax(0,1.05fr)]">
        <div className="flex min-w-0 flex-col gap-4">
          <EditorBriefCard brief={draft.brief} busy={calls.redraft.isLoading} canRedraft needs={needs} onChange={(brief) => setDraft((d) => ({ ...d, brief }))} onRedraft={redraft} />
          <EditorFormatCard
            draft={draft} durations={chosen?.durations ?? []} footageSlots={videoSlotsOf(chosen?.footage_slots ?? [], imageSlots)}
            footage={footage.data?.kinds.video} post={post} canEdit slides={slidesOf(draft)}
            onChange={setDraft} onFormat={(format) => setDraft((d) => withFormat(d, format, channels))}
            onSlides={(n) => setDraft((d) => withSlides(d, n))}
          />
          <EditorChannelsCard
            channels={channels} loading={channelsLoading} draft={draft} templateSizes={chosen?.sizes ?? []} onRefetch={refetchChannels}
            onTick={(channel, on) => { touchChannels(); setDraft((d) => withChannelTicked(d, channel, on)) }}
            onStory={(channel, on) => { touchChannels(); setDraft((d) => withStory(d, channel, on)) }}
            onOptions={(toolkit, options) => setDraft((d) => ({ ...d, options: { ...d.options, [toolkit]: options } }))}
          />
          {draft.format !== 'text' && (
            <EditorLookCard
              templates={templates.data ?? []} loading={templates.isLoading} chosen={draft.templateId} onPick={pickTemplate} busy={mediaBusy}
              photoSpots={photoSpots}
              onUpload={(file, spot) => calls.upload.mutate({ ...payload(), file, spot }, opened)}
              onLibrary={(item, spot) => calls.pick.mutate({ ...payload(), item, spot }, opened)}
              ai={{
                post, imageSlots, images: footage.data?.kinds.image, aiBusy: calls.aiMake.isLoading || calls.aiPick.isLoading,
                onAiMake: (imageSlot, prompt) => calls.aiMake.mutate({ ...payload(), imageSlot, prompt }, opened),
                onAiPick: (slot, name) => post && calls.aiPick.mutate({ postId: post.id, slot, name }),
              }}
            />
          )}
          {draft.format !== 'text' && <EditorAiLabelCard post={post} role={role} />}
          <EditorClaimsCard
            schema={chosen ? schemaAtLength(chosen.variables_schema, chosen.fields_cut_out, draft.lengthSeconds) : null}
            variables={draft.variables} sources={draft.sources} examples={chosen?.sample_data}
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
