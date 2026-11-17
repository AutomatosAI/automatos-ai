'use client'

/**
 * PRD-251D US-D203 — a post read as it is. The post page shows it to a role that cannot author
 * (a viewer, for every status) and for a post the editor does not hold (one being made,
 * publishing, posted, partly posted or archived); the Queue shows it to a person who may not
 * approve. It replaces the older detail pane (socials-post-detail.tsx) with the Studio's own
 * pieces:
 *
 * - the title, the status and the brief, and for a role that only reads, that it only reads;
 * - the post: its copy, each channel's copy, its rendered files and every claim with its source
 *   (socials-post-evidence.tsx), and the voice it is spoken with;
 * - its Activity card (socials-editor-activity.tsx): an approval an edit reset, the render lines
 *   (post-notices.tsx), each channel's receipt with Retry, the publish controls and the history.
 *
 * It offers nothing a role may not do: the receipts and the publish controls check the role
 * themselves, and the voice is read, never chosen, here.
 */
import { Badge } from '@/components/ui/badge'
import type { Workspace } from '@/components/workspace-provider'
import type { SocialPost } from '@/lib/api-client'
import { SocialsPostEvidence } from '../socials-post-evidence'
import { SOCIAL_STATUS_LABELS, canAuthorPosts, speaksAScript } from '../socials-status'
import { SocialsVoicePicker } from '../socials-voice-picker'
import { EditorCard, Hint } from './editor-ui'
import { SocialsEditorActivity } from './socials-editor-activity'

export const READ_ONLY_ROLE = 'Your role can read posts but not change them.'
export const POST_CARD = 'The post'
export const NO_COPY = 'No copy yet.'
const SECTION_TITLE = 'm-0 text-xs font-semibold uppercase tracking-wide text-muted-foreground'

interface SocialsPostViewProps {
  post: SocialPost
  role: Workspace['role']
}

/** The base copy every channel starts from: the channels below show what each one publishes. */
function BaseCopy({ post }: { post: SocialPost }) {
  return (
    <section aria-label="Copy" className="flex flex-col gap-1.5">
      <h3 className={SECTION_TITLE}>Copy</h3>
      <p className="m-0 whitespace-pre-wrap text-sm text-foreground">{post.copy?.base || NO_COPY}</p>
    </section>
  )
}

export function SocialsPostView({ post, role }: SocialsPostViewProps) {
  return (
    <article aria-label={`Post: ${post.title}`} className="flex min-w-0 flex-col gap-5">
      <header className="flex min-w-0 flex-col gap-1.5">
        <div className="flex flex-wrap items-center gap-3">
          <h2 className="m-0 break-words font-serif text-[26px] font-normal leading-[1.1] text-foreground">{post.title}</h2>
          <Badge variant="outline" className="h-6 rounded-full px-2.5" data-testid="socials-post-status">
            {SOCIAL_STATUS_LABELS[post.status]}
          </Badge>
        </div>
        {post.brief && <Hint>{post.brief}</Hint>}
        {!canAuthorPosts(role) && <Hint>{READ_ONLY_ROLE}</Hint>}
      </header>
      <div className="grid items-start gap-5 xl:grid-cols-[minmax(0,1.05fr)_minmax(0,0.95fr)]">
        <EditorCard label={POST_CARD}>
          <BaseCopy post={post} />
          <SocialsPostEvidence post={post} />
          {speaksAScript(post) && <SocialsVoicePicker post={post} editable={false} />}
        </EditorCard>
        <SocialsEditorActivity post={post} role={role} />
      </div>
    </article>
  )
}
