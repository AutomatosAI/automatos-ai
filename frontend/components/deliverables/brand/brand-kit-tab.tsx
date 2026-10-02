'use client'

/**
 * PRD-251B US-B301 — the Brand kit tab: one brand kit, on its own Deliverables tab, for
 * Templates and Socials alike (it replaces the BrandKitDialog everywhere; Template Studio
 * and Socials link here). Always visible: everyone in the workspace reads it, and owners
 * and admins (workspace:manage) change it. The basics, the style references and what Auto
 * takes from them, and the AI tools.
 */
import { useWorkspace } from '@/components/workspace-provider'
import { useBrandStyle } from '@/hooks/use-brand-style'
import { canEditBrandKit } from '../socials/socials-status'
import { BrandAiTools } from './brand-ai-tools'
import { BrandKitBasics } from './brand-kit-basics'
import { BrandReferences } from './brand-references'
import { BrandStyleProfileCard } from './brand-style-profile'
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
      <BrandKitBasics form={form} canEdit={canEdit} />
      <div className="grid items-start gap-4 lg:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)]">
        <BrandReferences style={style} canEdit={canEdit} />
        <BrandStyleProfileCard style={style} canEdit={canEdit} />
      </div>
      <BrandAiTools canEdit={canEdit} />
    </div>
  )
}
