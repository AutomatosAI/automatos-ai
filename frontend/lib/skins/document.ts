// PRD-259 §4 — where a skin lives in this document: <html data-skin="<id>"> and the one
// <style id="automatos-skin"> element (the pre-paint script's element), plus the cached
// stylesheet the pre-paint script reads on the next load. The stylesheet comes from
// skinToCss only and is set as textContent, never innerHTML.

import { clearSkinCache, writeSkinCache } from './cache'
import { SKIN_DATA_ATTRIBUTE } from './css'
import { SKIN_STYLE_ELEMENT_ID } from './prepaint'
import type { ResolvedSkin } from './installed'

function skinStyleElement(): HTMLStyleElement {
  const existing = document.getElementById(SKIN_STYLE_ELEMENT_ID)
  if (existing instanceof HTMLStyleElement) return existing
  existing?.remove()
  const element = document.createElement('style')
  element.id = SKIN_STYLE_ELEMENT_ID
  document.head.appendChild(element)
  return element
}

/** Puts the resolved skin on the page and caches its stylesheet for the next first paint. */
export function applySkinToDocument(resolved: ResolvedSkin): void {
  if (typeof document === 'undefined') return
  document.documentElement.setAttribute(SKIN_DATA_ATTRIBUTE, resolved.id)
  const element = skinStyleElement()
  if (element.textContent !== resolved.css) element.textContent = resolved.css
  writeSkinCache({ id: resolved.id, version: resolved.skin.version, css: resolved.css })
}

/** Takes any skin off the page and out of the cache: the base style shows. */
export function removeSkinFromDocument(): void {
  if (typeof document === 'undefined') return
  document.documentElement.removeAttribute(SKIN_DATA_ATTRIBUTE)
  document.getElementById(SKIN_STYLE_ELEMENT_ID)?.remove()
  clearSkinCache()
}
