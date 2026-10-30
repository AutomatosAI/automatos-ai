'use client'

/**
 * PRD-261 — the hero's devices: the current view's web screenshot in a browser frame, and its
 * Automatos Go screenshot in a phone frame in front of it. A view with no phone screenshot (the
 * listing has none, storefront-views.ts) shows no phone frame: no device is drawn without a real
 * screen in it. These are the hero's own images, so they load at once (the strip's are lazy).
 */
import { Palette } from 'lucide-react'

import { cn } from '@/lib/utils'

import { PHONE_CAPTION, WEB_CAPTION, phoneScreenAlt, webScreenAlt } from './copy'
import type { StorefrontView } from './storefront-views'

const CAPTION = 'rounded-md bg-background/85 px-2 py-1 font-mono text-[11px] uppercase tracking-wider text-foreground'
const BROWSER_DOTS = ['first', 'second', 'third'] as const

function BrowserFrame({ src, alt }: { src: string | null; alt: string }) {
  return (
    <figure className="relative m-0 overflow-hidden rounded-xl border border-border bg-background shadow-2xl">
      <div className="flex h-7 items-center gap-1.5 border-b border-border bg-muted px-3" aria-hidden>
        {BROWSER_DOTS.map((dot) => (
          <span key={dot} className="h-2.5 w-2.5 rounded-full bg-border" />
        ))}
      </div>
      {src ? (
        // A theme file on the app's own origin (A-08b proxy); next/image would re-encode it.
        // eslint-disable-next-line @next/next/no-img-element
        <img
          key={src}
          src={src}
          alt={alt}
          data-testid="storefront-web-screen"
          className="block aspect-[16/10] w-full object-cover motion-safe:duration-300 motion-safe:animate-in motion-safe:fade-in"
        />
      ) : (
        <div className="flex aspect-[16/10] items-center justify-center bg-muted" aria-hidden>
          <Palette className="h-10 w-10 text-muted-foreground" />
        </div>
      )}
      <figcaption className={cn(CAPTION, 'absolute bottom-3 left-3')}>{WEB_CAPTION}</figcaption>
    </figure>
  )
}

function PhoneFrame({ src, alt }: { src: string; alt: string }) {
  return (
    <figure
      className="absolute bottom-0 right-0 m-0 aspect-[196/412] w-[30%] max-w-[196px] rounded-[2rem] border border-border bg-background p-1.5 shadow-2xl"
      data-testid="storefront-phone"
    >
      {/* eslint-disable-next-line @next/next/no-img-element -- a theme file on the app's own origin */}
      <img
        key={src}
        src={src}
        alt={alt}
        className="block h-full w-full rounded-[1.6rem] object-cover object-top motion-safe:duration-300 motion-safe:animate-in motion-safe:fade-in"
      />
      <figcaption className={cn(CAPTION, 'absolute -bottom-8 left-1/2 -translate-x-1/2 whitespace-nowrap')}>
        {PHONE_CAPTION}
      </figcaption>
    </figure>
  )
}

export function HeroDevices({ view, name }: { view: StorefrontView; name: string }) {
  return (
    <div className={cn('relative min-w-0 flex-[999_1_560px]', view.phone && 'pb-10 pr-[18%]')}>
      <BrowserFrame src={view.web} alt={webScreenAlt(name, view.label)} />
      {view.phone && <PhoneFrame src={view.phone} alt={phoneScreenAlt(name, view.label)} />}
    </div>
  )
}
