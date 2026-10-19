'use client'

/**
 * PRD-264 EUAI-08a (EU AI Act Art. 50(4)) — the composer's "Realistic AI people, places,
 * products or voice: label as AI-generated". On by default whenever the post carries AI
 * media (AI footage, AI stills or a synthetic voice); the customer may switch it off for
 * this post, and the card says why that is theirs to decide. While it is on, the video
 * shows "AI-generated" at its start and at an interval, and every caption starts with it.
 *
 * The backend owns the rule and the words (GET /ai-labels, `deepfake_label`); the switch
 * calls PUT /ai-label. The setting is approved content: an approved post goes back for
 * approval. A post exists before the setting does, so a new post shows a hint instead.
 */
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import type { Workspace } from '@/components/workspace-provider'
import type { SocialPost } from '@/lib/api-client'
import { useSetSocialPostAiLabel, useSocialPostAiLabels } from '@/hooks/use-socials-ai-labels'
import { canAuthorPosts } from '../socials-status'
import { EditorCard, Hint } from './editor-ui'

export const AI_LABEL_CARD = 'AI-generated label'
export const AI_LABEL_SETTING = 'Realistic AI people, places, products or voice: label as AI-generated'
export const AI_LABEL_NEW_POST = 'Save the draft first: the label is on by default whenever the post carries AI media.'
export const AI_LABEL_DEFAULT = 'On by default: this post carries AI media.'
export const AI_LABEL_NO_AI = 'Off by default: this post carries no AI media.'
const SWITCH_ID = 'socials-editor-ai-label'

interface EditorAiLabelCardProps {
  post: SocialPost | null
  role: Workspace['role']
}

function AiLabelSwitch({ post, role }: { post: SocialPost; role: Workspace['role'] }) {
  const labels = useSocialPostAiLabels(post, { always: true })
  const set = useSetSocialPostAiLabel()
  const label = labels.data?.deepfake_label
  if (!label) return <Hint>{labels.isError ? 'The AI-generated label could not be read.' : 'Reading the AI-generated label…'}</Hint>
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-3">
        <Switch
          id={SWITCH_ID}
          checked={label.on}
          disabled={!canAuthorPosts(role) || set.isLoading}
          aria-describedby={`${SWITCH_ID}-reason`}
          onCheckedChange={(on) => set.mutate({ postId: post.id, on })}
        />
        <Label htmlFor={SWITCH_ID} className="text-sm text-foreground">{label.name || AI_LABEL_SETTING}</Label>
      </div>
      <Hint>{label.default ? AI_LABEL_DEFAULT : AI_LABEL_NO_AI}</Hint>
      <p id={`${SWITCH_ID}-reason`} className="m-0 text-[12.5px] leading-[1.45] text-foreground" data-testid="socials-ai-label-reason">
        {label.on ? `Switching it off: ${label.reason.charAt(0).toLowerCase()}${label.reason.slice(1)}` : label.reason}
      </p>
      <Hint>{label.line}</Hint>
    </div>
  )
}

export function EditorAiLabelCard({ post, role }: EditorAiLabelCardProps) {
  return (
    <EditorCard label={AI_LABEL_CARD}>
      {post ? <AiLabelSwitch post={post} role={role} /> : <Hint>{AI_LABEL_NEW_POST}</Hint>}
    </EditorCard>
  )
}
