'use client'

/**
 * PRD-251 S2.3 (US-206) — what an approver approves, shown as it will publish:
 * the channels with each one's copy (the base, or the channel's own text), the
 * EXACT rendered files (GET /api/socials/posts/{id}/media, through the shared
 * FilePreview), and every claim with the source it cites, or a red Unsourced
 * chip (D7).
 */
import { FilePreview, inferPreviewType } from '@/components/widgets/FileWidget/FilePreview'
import { badgeVariants } from '@/components/ui/badge'
import type { SocialPost } from '@/lib/api-client'
import { useSocialPostMedia } from '@/hooks/use-socials-api'
import { channelCopies, claimRows } from './socials-review'
import { channelLabel } from './socials-status'

const SECTION_TITLE = 'text-xs font-semibold uppercase tracking-wide text-muted-foreground'

export function UnsourcedChip() {
  return <span className={badgeVariants({ variant: 'destructive' })}>Unsourced</span>
}

function SocialsPostChannels({ post }: { post: SocialPost }) {
  const channels = channelCopies(post)
  return (
    <section aria-label="Channels" className="space-y-2">
      <h4 className={SECTION_TITLE}>Channels</h4>
      {channels.length === 0 ? (
        <p className="text-sm text-muted-foreground">No channels chosen yet.</p>
      ) : (
        <ul className="space-y-2">
          {channels.map((channel) => (
            <li key={channel.toolkit} className="rounded-lg border border-border/60 px-3 py-2">
              <p className="text-sm font-medium text-foreground">
                {channelLabel(channel.toolkit)}
                <span className="font-normal text-muted-foreground"> · {channel.kinds.join(', ')}</span>
              </p>
              <p className="whitespace-pre-wrap text-sm text-foreground">{channel.text || 'No copy yet.'}</p>
              {!channel.overridden && <p className="text-xs text-muted-foreground">The base copy</p>}
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}

function SocialsPostMedia({ post }: { post: SocialPost }) {
  const hasMedia = Object.keys(post.media ?? {}).length > 0
  const { data, isLoading, isError } = useSocialPostMedia(post.id, post.content_hash, hasMedia)
  if (!hasMedia) return null
  return (
    <section aria-label="Media" className="space-y-2">
      <h4 className={SECTION_TITLE}>Media</h4>
      {isLoading && <p className="text-sm text-muted-foreground">Loading the rendered files…</p>}
      {isError && <p className="text-sm text-destructive">The rendered files could not be loaded.</p>}
      <div className="grid gap-3 sm:grid-cols-2">
        {(data ?? []).map((file) => (
          <figure key={`${file.aspect}-${file.deliverable_id}`} className="space-y-1">
            {file.url ? (
              <div className="h-64 overflow-hidden rounded-lg border border-border/60">
                <FilePreview
                  url={file.url}
                  filename={file.name ?? undefined}
                  previewType={inferPreviewType(file.name ?? '', file.content_type ?? '')}
                />
              </div>
            ) : (
              <p className="text-sm text-destructive">{file.error ?? 'This file is missing.'}</p>
            )}
            <figcaption className="text-xs text-muted-foreground">{file.aspect}</figcaption>
          </figure>
        ))}
      </div>
    </section>
  )
}

function SocialsPostClaims({ post }: { post: SocialPost }) {
  const claims = claimRows(post)
  if (claims.length === 0) return null
  return (
    <section aria-label="Claims" className="space-y-2">
      <h4 className={SECTION_TITLE}>Claims and sources</h4>
      <ul className="space-y-1.5">
        {claims.map((claim) => (
          <li key={claim.name} className="text-sm" data-testid={`socials-claim-${claim.name}`}>
            <span className="font-medium text-foreground">{claim.name}</span>
            {claim.value && <span className="text-foreground">: {claim.value}</span>}{' '}
            {claim.source ? (
              <span className="text-muted-foreground">
                — {claim.source.kind} {claim.source.ref}
                {claim.source.as_of ? `, as of ${claim.source.as_of}` : ''}
              </span>
            ) : (
              <UnsourcedChip />
            )}
          </li>
        ))}
      </ul>
    </section>
  )
}

/** Channels, media and claims: everything an approval binds to (D6, D7). */
export function SocialsPostEvidence({ post }: { post: SocialPost }) {
  return (
    <div className="space-y-4">
      <SocialsPostChannels post={post} />
      <SocialsPostMedia post={post} />
      <SocialsPostClaims post={post} />
    </div>
  )
}
