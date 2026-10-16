'use client'

import { useState, useEffect, useRef } from 'react'
import { useTheme } from 'next-themes'
import { Sun, Moon, Monitor, BookOpen, LayoutTemplate } from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { AddSkinItem, InstalledSkinItems, SkinFileInput } from '@/components/ui/skin-menu-items'
import { useUiStyle } from '@/contexts/ui-style-context'
import { useSkinImport } from '@/hooks/use-skin-import'
import { isUiStyle } from '@/lib/ui-style'

/**
 * PRD-244 D1 — two axes in one menu: Style (Classic | Studio) and Tone
 * (Light | Dark | System). Picking on one axis never changes the other.
 *
 * PRD-259 §4 — the Style axis also lists the workspace's installed skins and
 * "Add a skin…". A single-tone skin locks the tone (ThemeProvider's
 * forcedTheme): the other tones are disabled and the user's own stored tone
 * comes back when they leave the skin.
 */
export function ThemeToggle() {
  const { setTheme: storeTone, theme: chosenTone, forcedTheme } = useTheme()
  const theme = forcedTheme ?? chosenTone
  const toneLocked = (tone: string) => forcedTheme !== undefined && tone !== forcedTheme
  // While a skin locks the tone, picking it again must not overwrite the user's own.
  const setTheme = (tone: string) => { if (forcedTheme === undefined) storeTone(tone) }
  const { style, setStyle, installedSkins } = useUiStyle()
  const { importing, importFile } = useSkinImport()
  const fileInput = useRef<HTMLInputElement>(null)
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  if (!mounted) {
    return (
      <Button variant="ghost" size="icon" disabled>
        <Sun className="h-4 w-4" />
      </Button>
    )
  }

  const ToneIcon = theme === 'light' ? Sun : theme === 'dark' ? Moon : Monitor

  return (
    <>
      <SkinFileInput ref={fileInput} onFile={(file) => { void importFile(file) }} />
      <DropdownMenu onOpenChange={(open) => { if (open) void installedSkins.reload() }}>
        <DropdownMenuTrigger asChild>
          <Button variant="ghost" size="icon" className="hover:bg-secondary/80" aria-label="Appearance">
            <ToneIcon className="h-4 w-4" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="min-w-[160px]">
          <DropdownMenuLabel>Style</DropdownMenuLabel>
          <DropdownMenuRadioGroup value={style} onValueChange={(v) => { if (isUiStyle(v)) setStyle(v) }}>
            <DropdownMenuRadioItem value="classic">
              <LayoutTemplate className="mr-2 h-4 w-4" />
              Classic
            </DropdownMenuRadioItem>
            <DropdownMenuRadioItem value="studio">
              <BookOpen className="mr-2 h-4 w-4" />
              Studio
            </DropdownMenuRadioItem>
            <InstalledSkinItems skins={installedSkins.skins} />
          </DropdownMenuRadioGroup>
          <AddSkinItem importing={importing} onChoose={() => fileInput.current?.click()} />
          <DropdownMenuSeparator />
          <DropdownMenuLabel>Tone</DropdownMenuLabel>
          <DropdownMenuRadioGroup value={theme ?? 'system'} onValueChange={setTheme}>
            <DropdownMenuRadioItem value="light" disabled={toneLocked('light')}>
              <Sun className="mr-2 h-4 w-4" />
              Light
            </DropdownMenuRadioItem>
            <DropdownMenuRadioItem value="dark" disabled={toneLocked('dark')}>
              <Moon className="mr-2 h-4 w-4" />
              Dark
            </DropdownMenuRadioItem>
            <DropdownMenuRadioItem value="system" disabled={toneLocked('system')}>
              <Monitor className="mr-2 h-4 w-4" />
              System
            </DropdownMenuRadioItem>
          </DropdownMenuRadioGroup>
        </DropdownMenuContent>
      </DropdownMenu>
    </>
  )
}
