/**
 * F378 (night 11, 7 Oct) — the composer asks the owner for what the brief does not give
 * (`questions` on its proposal), and the editor shows it as one line, "Auto needs: …".
 */
import { describe, it, expect } from 'vitest'

import { autoNeeds, draftFromProposal } from '@/components/deliverables/socials/socials-composer-model'
import type { SocialComposeProposal } from '@/lib/api-client'

const PROPOSAL: SocialComposeProposal = {
  title: 'Our numbers', copy: { base: 'We sold 412 bags.', channels: { twitter: '412 bags.' } }, format: 'image',
  template_id: null, template: null, variables: {}, sources: {}, channels: ['twitter'], warnings: [],
  questions: ['Source line', ' ', "A photo for 'Photo'"],
}

describe("Auto's questions for the owner", () => {
  it('reads as one line, and as nothing when there are none', () => {
    expect(autoNeeds(PROPOSAL.questions)).toBe("Auto needs: Source line; A photo for 'Photo'")
    expect(autoNeeds([])).toBeNull()
    expect(autoNeeds(undefined)).toBeNull()
  })

  it("rides into the composer's draft with the copy in the shape a save takes", () => {
    const draft = draftFromProposal('Three numbers', PROPOSAL, [])
    expect(draft.questions).toEqual(PROPOSAL.questions)
    expect(draft.perChannel).toEqual({ twitter: '412 bags.' })
  })
})
