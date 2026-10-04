'use client'

/**
 * PRD-251C (C8, US-C406) — Brand kit, Voice: how the owner rewrites Auto. When a person
 * approves a post whose copy they changed from Auto's draft, the pair is kept, and Auto's
 * composer writes like the approved versions. Owners and admins remove any example; a removed
 * one is never used again. Hidden when Socials is off (the list answers 404 then).
 */
import { Button } from '@/components/ui/button'
import { useRemoveVoiceExample, useVoiceExamples } from '@/hooks/use-socials-results'

export const VOICE_EXAMPLES_NOTE =
  "When you rewrite Auto's copy before approving a post, the two versions are kept here, and Auto writes like your versions."
export const NO_VOICE_EXAMPLES = "None yet: approve a post whose copy you changed from Auto's draft, and it shows here."

export function BrandVoiceExamples({ canEdit }: { canEdit: boolean }) {
  const examples = useVoiceExamples()
  const remove = useRemoveVoiceExample()
  if (examples.isError) return null
  const rows = examples.data?.examples ?? []
  return (
    <section aria-label="Voice examples" className="flex flex-col gap-3 rounded-xl border border-border bg-card p-4">
      <div className="flex flex-col gap-1">
        <h3 className="m-0 text-base font-semibold text-foreground">Voice: how you rewrite Auto</h3>
        <p className="m-0 text-sm text-muted-foreground">{VOICE_EXAMPLES_NOTE}</p>
      </div>
      {rows.length === 0 ? (
        <p className="m-0 text-sm text-muted-foreground">{examples.isLoading ? 'Loading…' : NO_VOICE_EXAMPLES}</p>
      ) : (
        <ul className="m-0 flex list-none flex-col gap-2.5 p-0">
          {rows.map((example) => (
            <li key={example.id} aria-label={example.approved} className="flex flex-col gap-1.5 rounded-lg border border-border p-3">
              <p className="m-0 text-[13px] text-muted-foreground"><span className="font-medium">Auto wrote:</span> {example.draft}</p>
              <p className="m-0 text-[13px] text-foreground"><span className="font-medium">You approved:</span> {example.approved}</p>
              {canEdit && (
                <Button type="button" size="sm" variant="ghost" className="self-end" disabled={remove.isLoading}
                  onClick={() => remove.mutate(example.id)}>
                  Remove
                </Button>
              )}
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}
