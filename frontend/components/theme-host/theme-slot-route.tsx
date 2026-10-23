'use client'

import { Suspense } from 'react'
import { notFound } from 'next/navigation'

import { registeredComponent, useActiveThemeSlots } from './theme-slot-registrations'

/**
 * PRD-261 A-04b — a route a theme registered (registerRoute), served at /x/<route>
 * (app/x/[...path]/page.tsx renders this inside MainLayout, full bleed). A route has no core
 * version: when the active theme registered none by this name, or there is no theme, the address
 * is the 404. While an installed theme loads (its routes are not known yet) nothing renders. A
 * route renders at any width: it is the theme's own page, and decides itself what to do on a
 * narrow screen (installed Play's Conversations, for one, sends a phone to /chat).
 */
export function ThemeSlotRoute({ route }: { readonly route: string }) {
  const { layout, registrations } = useActiveThemeSlots()
  if (registrations === null && layout === 'pending') return null
  const Route = registeredComponent(registrations?.routes.get(route))
  if (Route === null) notFound()
  return (
    <Suspense fallback={null}>
      <Route />
    </Suspense>
  )
}
