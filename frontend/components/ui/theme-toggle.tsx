'use client'

import { useState, useEffect } from 'react'
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
import { isUiStyle } from '@/lib/ui-style'

/** PRD-260 K9: Play's look editor ("Make it yours"). */
const LOOK_EDITOR_HREF = '/looks'

/**
 * PRD-260 K9: the look editor, offered only while Play is the style — the
 * menu never switches style by surprise.
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
  const { setTheme, theme } = useTheme()
  const { style, setStyle, isPlay } = useUiStyle()
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
        {isPlay && <CustomiseLookItem />}
        <DropdownMenuSeparator />
        <DropdownMenuLabel>Tone</DropdownMenuLabel>
        <DropdownMenuRadioGroup value={theme ?? 'system'} onValueChange={setTheme}>
          <DropdownMenuRadioItem value="light">
            <Sun className="mr-2 h-4 w-4" />
            Light
          </DropdownMenuRadioItem>
          <DropdownMenuRadioItem value="dark">
            <Moon className="mr-2 h-4 w-4" />
            Dark
          </DropdownMenuRadioItem>
          <DropdownMenuRadioItem value="system">
            <Monitor className="mr-2 h-4 w-4" />
            System
          </DropdownMenuRadioItem>
        </DropdownMenuRadioGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
