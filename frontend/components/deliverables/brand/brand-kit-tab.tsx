'use client'

/**
 * PRD-251B US-B301 — the Brand kit tab: one brand kit, on its own Deliverables tab, for
 * Templates and Socials alike (it replaces the BrandKitDialog everywhere; Template Studio
 * and Socials link here). Always visible: everyone in the workspace reads it, and owners
 * and admins (workspace:manage) change it. The brand board (PRD-255 US-010: the kit on one
 * page, drawn again after each save, to download as a PDF or a PNG), the basics, the design
 * system (PRD-255: colour roles, type, spacing and logo, logo variants, locale), the style references and what Auto
 * takes from them, the owner's voice examples (PRD-251C US-C406), and the AI tools.
 */
import { useWorkspace } from '@/components/workspace-provider'
import { useBrandStyle } from '@/hooks/use-brand-style'
import { canEditBrandKit } from '../socials/socials-status'
import { BrandAiTools } from './brand-ai-tools'
import { BrandBoardPreview } from './brand-board-preview'
import { BrandKitBasics } from './brand-kit-basics'
import { BrandKitDesign } from './brand-kit-design'
import { BrandReferences } from './brand-references'
import { BrandStyleProfileCard } from './brand-style-profile'
import { BrandVoiceExamples } from './brand-voice-examples'
import { useBrandKitForm } from './use-brand-kit-form'

export const READ_ONLY_NOTE = 'Only owners and admins change the brand kit.'

export function BrandKitTab() {
  const { workspace } = useWorkspace()
  const canEdit = canEditBrandKit(workspace?.role)
  const form = useBrandKitForm()
  const style = useBrandStyle()
  return (
    <div className="flex flex-col gap-5">
      <div>
        <h2 className="text-xl font-bold text-foreground">Brand kit</h2>
        <p className="text-sm text-muted-foreground">
          Applied to every document and social post rendered from a template: logo, colours, fonts, voice, and the style Auto follows in every image it asks for.
        </p>
        {!canEdit && <p className="mt-1 text-xs text-muted-foreground">{READ_ONLY_NOTE}</p>}
      </div>
      <BrandBoardPreview changes={form.boardVersion} />
      <BrandKitBasics form={form} canEdit={canEdit} />
      <BrandKitDesign form={form} canEdit={canEdit} />
      <div className="grid items-start gap-4 lg:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)]">
        <BrandReferences style={style} canEdit={canEdit} />
        <BrandStyleProfileCard style={style} canEdit={canEdit} />
      </div>
      <BrandVoiceExamples canEdit={canEdit} />
      <BrandAiTools canEdit={canEdit} />
    </div>
  )
}
