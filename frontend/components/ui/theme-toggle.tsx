'use client'

import { useState, useEffect, useId } from 'react'
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
import { useIsTabletOrBelow } from '@/hooks/use-mobile'
import { isUiStyle } from '@/lib/ui-style'

/** PRD-260 K9: Play's look editor ("Make it yours"). */
const LOOK_EDITOR_HREF = '/looks'

/** PRD-260 K4: under Play the look's ground sets the tone, so the Tone items rest. */
const TONE_SET_BY_LOOK = 'Set by your look'

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
 * PRD-260 K9: the look editor, offered only while Play is the style — the
 * menu never switches style by surprise — and from 1024 px up, where /looks opens (D7).
 */
function CustomiseLookItem() {
  return (
    <DropdownMenuItem asChild>
      <Link href={LOOK_EDITOR_HREF}>
        <Palette className="mr-2 h-4 w-4" />
        Customise look…
      </Link>
    </DropdownMenuItem>
  )
}

/**
 * PRD-244 D1 — two axes in one menu: Style (Classic | Studio | Play) and Tone
 * (Light | Dark | System). Picking on one axis never changes the other.
 */
export function ThemeToggle() {
  const { setTheme: setStoredTone, theme: storedTone, forcedTheme } = useTheme()
  // The tone in effect: under Play the look forces one and keeps the stored one for later.
  const theme = forcedTheme ?? storedTone
  const { style, setStyle, isPlay } = useUiStyle()
  // PRD-260 K4: the look paints at every width, so it sets the tone at every width; its
  // editor, /looks, opens only from 1024 px up (D7), so 'Customise look…' shows only there.
  const narrow = useIsTabletOrBelow()
  const toneLocked = isPlay
  const offersLookEditor = isPlay && !narrow
  const [mounted, setMounted] = useState(false)
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
    <DropdownMenu>
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
          <DropdownMenuRadioItem value="play">
            <Sparkles className="mr-2 h-4 w-4" />
            Play
          </DropdownMenuRadioItem>
        </DropdownMenuRadioGroup>
        {offersLookEditor && <CustomiseLookItem />}
        <DropdownMenuSeparator />
        <DropdownMenuLabel>Tone</DropdownMenuLabel>
        {toneLocked && <p id={toneNoteId} className="px-2 pb-1.5 text-xs text-muted-foreground">{TONE_SET_BY_LOOK}</p>}
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
