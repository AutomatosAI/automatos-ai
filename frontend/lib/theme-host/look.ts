/**
 * PRD-261 A-01 — applies a theme's look: its class on <html>, its one stylesheet, then its preset's
 * CSS variables (style.setProperty only). Returns the cleanup that removes all three. A failure
 * part-way removes what was applied, then rethrows.
 */
import type { LookPreset, SpikeManifest } from './types'

/** Marks the host's theme stylesheet, so the browser lane (and a cleanup) can find it. */
export const THEME_STYLE_LINK_ATTR = 'data-theme-host-style'

function waitForStylesheet(link: HTMLLinkElement, timeoutMs: number): Promise<void> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`the stylesheet did not load within ${timeoutMs} ms`)), timeoutMs)
    link.addEventListener('load', () => {
      clearTimeout(timer)
      resolve()
    }, { once: true })
    link.addEventListener('error', () => {
      clearTimeout(timer)
      reject(new Error(`the stylesheet failed to load: ${link.href}`))
    }, { once: true })
  })
}

function stylesheetLink(doc: Document, href: string): HTMLLinkElement {
  const link = doc.createElement('link')
  link.rel = 'stylesheet'
  link.href = href
  link.setAttribute(THEME_STYLE_LINK_ATTR, '')
  return link
}

export async function applyLook(
  manifest: SpikeManifest,
  preset: LookPreset,
  doc: Document,
  timeoutMs: number,
): Promise<() => void> {
  const root = doc.documentElement
  const link = stylesheetLink(doc, manifest.styleSheet)
  const variables = Object.entries(preset.vars)
  const cleanup = (): void => {
    root.classList.remove(manifest.htmlClass)
    link.remove()
    variables.forEach(([name]) => root.style.removeProperty(name))
  }
  try {
    root.classList.add(manifest.htmlClass)
    const loaded = waitForStylesheet(link, timeoutMs)
    doc.head.appendChild(link)
    await loaded
    variables.forEach(([name, value]) => root.style.setProperty(name, value))
    return cleanup
  } catch (error) {
    cleanup()
    throw error
  }
}
