/**
 * A browser window of a given width for jsdom, which has no layout and no media
 * queries: window.matchMedia answers `(max-width: Npx)` and `(min-width: Npx)`
 * as a window that wide would, and no other query (reduced motion, colour
 * scheme) matches. Resizing tells every listener whose query flips, as a
 * browser's `change` event does. P260-FX-25 (M69): a theme's width source
 * (useThemeLayout) and useIsTabletOrBelow both read it.
 */

/** A phone, below Play's 1024 px (D7). */
export const PHONE_WIDTH_PX = 390
/** A laptop, at Play's width. */
export const LAPTOP_WIDTH_PX = 1280

const WIDTH_QUERY = /^\((max|min)-width:\s*(\d+)px\)$/

type Listener = (event: MediaQueryListEvent) => void

interface QueryList {
  readonly media: string
  readonly listeners: Set<Listener>
}

export interface WindowWidth {
  /** Every query asked, in order (a hook asks again on each read). */
  readonly asked: readonly string[]
  /** The window takes this width; every listener whose query flips hears the change. */
  resize(width: number): void
  /** window.matchMedia as it was. */
  restore(): void
}

function matchesAt(media: string, width: number): boolean {
  const match = WIDTH_QUERY.exec(media.trim())
  if (!match) return false
  const limit = Number(match[2])
  return match[1] === 'max' ? width <= limit : width >= limit
}

function mediaQueryList(list: QueryList, width: () => number): MediaQueryList {
  const add = (_type: string, listener: Listener) => list.listeners.add(listener)
  const remove = (_type: string, listener: Listener) => list.listeners.delete(listener)
  return {
    media: list.media,
    get matches() {
      return matchesAt(list.media, width())
    },
    onchange: null,
    addEventListener: add,
    removeEventListener: remove,
    addListener: (listener: Listener) => add('change', listener),
    removeListener: (listener: Listener) => remove('change', listener),
    dispatchEvent: () => false,
  } as unknown as MediaQueryList
}

/** Puts a window this wide in place of window.matchMedia until restore(). */
export function stubWindowWidth(initialWidth: number): WindowWidth {
  const original = Object.getOwnPropertyDescriptor(window, 'matchMedia')
  const lists = new Map<string, QueryList>()
  const asked: string[] = []
  let width = initialWidth

  const matchMedia = (media: string): MediaQueryList => {
    asked.push(media)
    const list = lists.get(media) ?? { media, listeners: new Set<Listener>() }
    lists.set(media, list)
    return mediaQueryList(list, () => width)
  }
  Object.defineProperty(window, 'matchMedia', { configurable: true, writable: true, value: matchMedia })

  return {
    asked,
    resize(next: number) {
      const before = width
      width = next
      for (const { media, listeners } of Array.from(lists.values())) {
        const matches = matchesAt(media, next)
        if (matches === matchesAt(media, before)) continue
        listeners.forEach((listener) => listener({ matches, media } as MediaQueryListEvent))
      }
    },
    restore() {
      if (original) Object.defineProperty(window, 'matchMedia', original)
      else Reflect.deleteProperty(window, 'matchMedia')
    },
  }
}
