'use client'

import { useState, useEffect, useId, useMemo } from 'react'
import Link from 'next/link'
import { useTheme } from 'next-themes'
import { Sun, Moon, Monitor, BookOpen, LayoutTemplate, Sparkles, Palette } from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { useUiStyle } from '@/contexts/ui-style-context'
import { useThemeLayout } from '@/hooks/use-studio-theme'
import {
  themeChoiceEntries,
  useChooseStyle,
  useWantInstalledThemes,
  type ThemeChoiceEntry,
} from '@/lib/theme-host/choice-sync'
import type { ThemeTonePolicy } from '@/lib/theme-host/registry-types'
import { LOOKS_HREF } from '@/lib/ticket-links'
import { isUiStyle } from '@/lib/ui-style'

/** PRD-260 K4: under a theme whose look's ground sets the tone, the Tone items rest. */
const TONE_SET_BY_LOOK = 'Set by your look'
/** PRD-261: under a theme that always paints one tone, the Tone items rest too. */
const TONE_SET_BY_THEME = 'Set by your theme'

/** The note over resting Tone items, or null when the user sets the tone (no theme, or a 'user' policy). */
function toneNoteFor(policy: ThemeTonePolicy | null): string | null {
  if (policy === null || policy === 'user') return null
  return policy === 'from-look' ? TONE_SET_BY_LOOK : TONE_SET_BY_THEME
}

/** A resting Tone item reads as unavailable; it keeps its focus highlight. */
const TONE_ITEM_CLASS = 'aria-disabled:opacity-50'

/** Radix closes the menu after a select unless the item's select event is prevented. */
function keepMenuOpen(event: Event): void {
  event.preventDefault()
}

/**
 * PRD-260 K4, M85: a resting Tone item stays focusable and announced, described by
 * the 'Set by your look' note: aria-disabled rather than `disabled`, which would take
 * it out of the keyboard's reach. Choosing it changes nothing and keeps the menu open.
 */
function restingToneItem(noteId: string) {
  return { 'aria-disabled': true, 'aria-describedby': noteId, onSelect: keepMenuOpen } as const
}

/**
 * PRD-260 K9, PRD-261 PA-14: the look editor, offered only while a theme is the style (every
 * theme ships one) — the menu never switches style by surprise — and only where the theme's
 * layout renders (its entry's minWidth, D7), which is where /looks opens.
 */
function CustomiseLookItem() {
  return (
    <DropdownMenuItem asChild>
      <Link href={LOOKS_HREF}>
        <Palette className="mr-2 h-4 w-4" />
        Customise look…
      </Link>
    </DropdownMenuItem>
  )
}

/**
 * PRD-261 A-02b, A-02c: the theme registry's entries after Classic and Studio, then the workspace's
 * installed themes (`theme:install:<id>`), each install that nobody reviewed noted as such.
 */
function ThemeStyleItems({ entries }: { entries: readonly ThemeChoiceEntry[] }) {
  return (
    <>
      {entries.map((entry) => (
        <DropdownMenuRadioItem key={entry.style} value={entry.style}>
          <Sparkles className="mr-2 h-4 w-4" />
          {entry.name}
          {entry.note && <span className="ml-auto pl-3 text-xs text-muted-foreground">{entry.note}</span>}
        </DropdownMenuRadioItem>
      ))}
    </>
  )
}

/**
 * PRD-244 D1 — two axes in one menu: Style (Classic | Studio | the registry's themes, PRD-261
 * A-02b) and Tone (Light | Dark | System). Picking on one axis never changes the other.
 * PRD-261 A-02c: a style pick is stored on the server as the member's choice (choice-sync.ts).
 */
export function ThemeToggle() {
  const { setTheme: setStoredTone, theme: storedTone, forcedTheme } = useTheme()
  // The tone in effect: under a theme's look the look forces one and keeps the stored one for later.
  const theme = forcedTheme ?? storedTone
  const { style, activeTheme, installedThemes } = useUiStyle()
  const chooseStyle = useChooseStyle()
  const themeEntries = useMemo(() => themeChoiceEntries(installedThemes, style), [installedThemes, style])
  // PRD-260 K4: the look paints at every width, so it sets the tone at every width; its
  // editor, /looks, opens only where the theme's layout renders (D7), so 'Customise look…' shows only there.
  const layout = useThemeLayout()
  const toneNote = toneNoteFor(activeTheme?.tone ?? null)
  const toneLocked = toneNote !== null
  const offersLookEditor = activeTheme !== null && layout === 'theme'
  const [mounted, setMounted] = useState(false)
  // The workspace's installs are fetched only once the menu is first opened (choice-sync.ts).
  const [menuOpen, setMenuOpen] = useState(false)
  useWantInstalledThemes(menuOpen)
  const toneNoteId = useId()
  // Under the lock the Tone group has no handler at all, so no pick reaches the stored tone.
  const setTheme = toneLocked ? undefined : setStoredTone
  const toneItemLock = toneLocked ? restingToneItem(toneNoteId) : {}

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
    <DropdownMenu open={menuOpen} onOpenChange={setMenuOpen}>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" size="icon" className="hover:bg-secondary/80" aria-label="Appearance">
          <ToneIcon className="h-4 w-4" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="min-w-[160px]">
        <DropdownMenuLabel>Style</DropdownMenuLabel>
        <DropdownMenuRadioGroup value={style} onValueChange={(v) => { if (isUiStyle(v)) chooseStyle(v) }}>
          <DropdownMenuRadioItem value="classic">
            <LayoutTemplate className="mr-2 h-4 w-4" />
            Classic
          </DropdownMenuRadioItem>
          <DropdownMenuRadioItem value="studio">
            <BookOpen className="mr-2 h-4 w-4" />
            Studio
          </DropdownMenuRadioItem>
          <ThemeStyleItems entries={themeEntries} />
        </DropdownMenuRadioGroup>
        {offersLookEditor && <CustomiseLookItem />}
        <DropdownMenuSeparator />
        <DropdownMenuLabel>Tone</DropdownMenuLabel>
        {toneLocked && <p id={toneNoteId} className="px-2 pb-1.5 text-xs text-muted-foreground">{toneNote}</p>}
        <DropdownMenuRadioGroup value={theme ?? 'system'} onValueChange={setTheme}>
          <DropdownMenuRadioItem value="light" className={TONE_ITEM_CLASS} {...toneItemLock}>
            <Sun className="mr-2 h-4 w-4" />
            Light
          </DropdownMenuRadioItem>
          <DropdownMenuRadioItem value="dark" className={TONE_ITEM_CLASS} {...toneItemLock}>
            <Moon className="mr-2 h-4 w-4" />
            Dark
          </DropdownMenuRadioItem>
          <DropdownMenuRadioItem value="system" className={TONE_ITEM_CLASS} {...toneItemLock}>
            <Monitor className="mr-2 h-4 w-4" />
            System
          </DropdownMenuRadioItem>
        </DropdownMenuRadioGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
