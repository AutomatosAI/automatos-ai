'use client'

import { Suspense } from 'react'
import { notFound, redirect } from 'next/navigation'

import { useActiveTheme } from '@/hooks/use-studio-theme'
import { CHAT_HREF } from '@/lib/ticket-links'

import { registeredComponent, useActiveThemeSlots } from './theme-slot-registrations'

/**
 * PRD-261 A-04b — a route a theme registered (registerRoute), served at /x/<route>
 * (app/x/[...path]/page.tsx renders this inside MainLayout, full bleed). A route has no core
 * version. While an installed theme loads (its routes are not known yet) nothing renders. A
 * route renders at any width: it is the theme's own page, and decides itself what to do on a
 * narrow screen (installed Play's Conversations, for one, sends a phone to /chat).
 *
 * A route the active style doesn't serve sends the member to the chat: with no theme (Classic,
 * Studio, a theme that is off, failed or in safe mode), and under a theme that neither declares nor
 * registered it. That is a route another theme serves, such as Play's /x/conversations after a
 * switch to Studio or HUD, and the core can't tell it from a route no installed theme declares:
 * it knows the active theme's routes alone. The one 404 left is a route the active theme declares
 * in its theme.json but its code never registered, a defect of the theme (the browser lane's
 * @installed check looks for that 404 on every route a theme declares).
 */
export function ThemeSlotRoute({ route }: { readonly route: string }) {
  const { layout, registrations } = useActiveThemeSlots()
  const declared = useActiveTheme()?.routes.includes(route) ?? false
  if (registrations === null && layout === 'pending') return null
  const Route = registeredComponent(registrations?.routes.get(route))
  if (Route === null) return registrations !== null && declared ? notFound() : redirect(CHAT_HREF)
  return (
    <Suspense fallback={null}>
      <Route />
    </Suspense>
  )
}
