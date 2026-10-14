'use client'

/**
 * PRD-259 §4 — the Skins group of the Style dropdown: the workspace's installed skins as
 * Style choices beside Classic and Studio, and "Add a skin…". The file input sits OUTSIDE
 * the menu (the menu unmounts when it closes, which would drop the chooser's answer); the
 * menu item only opens it.
 */
import { forwardRef, type ChangeEvent } from 'react'
import { Palette, Plus } from 'lucide-react'
import { DropdownMenuItem, DropdownMenuLabel, DropdownMenuRadioItem } from '@/components/ui/dropdown-menu'
import { SKIN_FILE_ACCEPT, type InstalledSkin } from '@/lib/skins/installed'
import { skinStyle } from '@/lib/ui-style'

export const ADD_SKIN_LABEL = 'Add a skin…'
export const SKINS_LABEL = 'Skins'

/** The Skins label and one radio item per installed skin; sits inside the Style radio group. */
export function InstalledSkinItems({ skins }: { skins: readonly InstalledSkin[] }) {
  return (
    <>
      <DropdownMenuLabel>{SKINS_LABEL}</DropdownMenuLabel>
      {skins.map((skin) => (
        <DropdownMenuRadioItem key={skin.id} value={skinStyle(skin.id)}>
          <Palette className="mr-2 h-4 w-4" />
          <span className="truncate">{skin.name}</span>
        </DropdownMenuRadioItem>
      ))}
    </>
  )
}

interface AddSkinItemProps {
  importing: boolean
  onChoose: () => void
}

export function AddSkinItem({ importing, onChoose }: AddSkinItemProps) {
  return (
    <DropdownMenuItem disabled={importing} onSelect={onChoose} className="pl-8">
      <Plus className="mr-2 h-4 w-4" />
      {ADD_SKIN_LABEL}
    </DropdownMenuItem>
  )
}

interface SkinFileInputProps {
  onFile: (file: File) => void
}

/** The hidden .autoskin chooser. Cleared after each pick, so the same file can be picked again. */
export const SkinFileInput = forwardRef<HTMLInputElement, SkinFileInputProps>(function SkinFileInput({ onFile }, ref) {
  const onChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    event.target.value = ''
    if (file) onFile(file)
  }
  return (
    <input
      ref={ref}
      type="file"
      accept={SKIN_FILE_ACCEPT}
      className="hidden"
      tabIndex={-1}
      aria-label={ADD_SKIN_LABEL}
      data-testid="skin-file-input"
      onChange={onChange}
    />
  )
})
