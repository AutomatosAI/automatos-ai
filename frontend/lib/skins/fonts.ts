// PRD-259 §4, decision K2 — a skin's own woff2 faces, registered with the browser through the
// FontFace API. The asset route needs the bearer token, which a CSS url() cannot send, so the
// bytes come through the API client's seam (fetchApiFile: apiFileUrl(apiClient.getBaseUrl(), path)
// with apiClient.getAuthHeaders(), shared with fetchBrandFileObjectUrl) and go to FontFace as an ArrayBuffer:
// no @font-face, no url() in the stylesheet, and no blob: URL (the page's CSP font-src names
// none). The family is the one css.ts writes (skinFontFamily); a face the browser refuses, or
// a file the route won't serve, leaves the slot on its base font: logged, never thrown.

import { fetchApiFile } from '@/lib/api-file-fetch'

import { skinFontFamily } from './css'
import { FONT_FILE_PATTERN, FONT_SLOTS, SKIN_ID_PATTERN, type FontSlot, type Skin } from './schema'

const SKINS_API_PATH = '/api/skins'
const LOG_PREFIX = '[Skins]'

/** The skin asset route for one file, versioned so the immutable cache never serves an old one. */
export function skinAssetPath(skinId: string, file: string, version: string): string {
  return `${SKINS_API_PATH}/${encodeURIComponent(skinId)}/assets/${file}?v=${encodeURIComponent(version)}`
}

interface FontFile {
  readonly slot: FontSlot
  readonly file: string
}

/** The slots that name a woff2 in the package (a shipped face needs no registration). */
export function skinFontFiles(skin: Pick<Skin, 'fonts'>): FontFile[] {
  const fonts = skin.fonts ?? {}
  return FONT_SLOTS.flatMap((slot) => {
    const file = fonts[slot]
    return typeof file === 'string' && FONT_FILE_PATTERN.test(file) ? [{ slot, file }] : []
  })
}

function fontsApiAvailable(): boolean {
  return typeof FontFace !== 'undefined' && typeof document !== 'undefined' && Boolean(document.fonts)
}

async function loadFace(skinId: string, version: string, { slot, file }: FontFile): Promise<FontFace | null> {
  const family = skinFontFamily(skinId, slot)
  try {
    const blob = await fetchApiFile(skinAssetPath(skinId, file, version))
    if (blob === null) {
      console.warn(`${LOG_PREFIX} ${file} wasn't served; ${slot} keeps the base font.`)
      return null
    }
    return await new FontFace(family, await blob.arrayBuffer()).load()
  } catch (error) {
    console.warn(`${LOG_PREFIX} The browser could not load ${file}; ${slot} keeps the base font.`, error)
    return null
  }
}

/**
 * Registers the skin's woff2 faces and resolves to the function that removes them again
 * (call it when the skin changes or the provider unmounts). Abort `signal` when the skin
 * changes while the files are still loading: nothing is registered then.
 */
export async function registerSkinFonts(
  skin: Pick<Skin, 'fonts' | 'version'>,
  skinId: string,
  signal?: AbortSignal,
): Promise<() => void> {
  const files = skinFontFiles(skin)
  if (!SKIN_ID_PATTERN.test(skinId)) {
    console.warn(`${LOG_PREFIX} Not a skin id; no fonts registered.`)
    return () => undefined
  }
  if (files.length === 0 || !fontsApiAvailable()) return () => undefined
  const loaded = await Promise.all(files.map((file) => loadFace(skinId, skin.version, file)))
  if (signal?.aborted) return () => undefined
  const faces = loaded.filter((face): face is FontFace => face !== null)
  faces.forEach((face) => document.fonts.add(face))
  return () => faces.forEach((face) => document.fonts.delete(face))
}
