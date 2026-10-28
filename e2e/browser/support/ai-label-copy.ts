import type { Page } from '@playwright/test'

// PRD-264 EUAI-11a: reading the host's protected copy of the AI label in a real browser
// (frontend/components/ai-label/protected-caption.tsx). Its shadow root is closed, so no locator
// reaches the words: an init script, run before the app's own code, records each closed root
// attachShadow makes for the copy's host, and the page reads the newest one still on the page. The
// copy is click-through, which hit-testing skips, so the probe gives the copy pointer-events for the
// moment of elementsFromPoint and takes them back; the copy rebuilds itself after any such write,
// which is harmless here (and logs one "changed" warning).

/** frontend/components/ai-label/protected-caption-parts.ts AI_LABEL_CAPTION_ATTR. */
export const CAPTION_SELECTOR = '[data-ai-label-caption]'
const ROOTS_KEY = '__aiLabelCaptionRoots'

export interface CopyState {
  readonly text: string | null
  /** In the browser's top layer (`:popover-open`). */
  readonly open: boolean
  readonly inViewport: boolean
  /** The browser paints it: not hidden, faded or skipped (checkVisibility). */
  readonly painted: boolean
  /** Hit-testing at its centre finds the copy's host first: nothing is drawn over it. */
  readonly topmost: boolean
}

/** Records the copy's closed shadow roots for copyState; add it before the first navigation. */
export async function recordCopyRoots(page: Page): Promise<void> {
  await page.addInitScript(({ key, selector }) => {
    const attach = Element.prototype.attachShadow
    const roots: ShadowRoot[] = []
    Element.prototype.attachShadow = function (this: Element, init: ShadowRootInit): ShadowRoot {
      const root = attach.call(this, init)
      if (this.matches(selector)) roots.push(root)
      return root
    }
    Object.defineProperty(window, key, { value: roots })
  }, { key: ROOTS_KEY, selector: CAPTION_SELECTOR })
}

/** The copy on the page now, or null when there is none. */
export async function copyState(page: Page): Promise<CopyState | null> {
  return page.evaluate((key) => {
    const roots = (window as unknown as Record<string, ShadowRoot[] | undefined>)[key] ?? []
    const root = [...roots].reverse().find((candidate) => candidate.host.isConnected)
    const caption = root?.querySelector<HTMLElement>('[role="note"]')
    if (!root || !caption) return null
    const host = root.host as HTMLElement
    const box = caption.getBoundingClientRect()
    const inViewport = box.width > 0 && box.height > 0 && box.left >= 0 && box.top >= 0 &&
      box.right <= window.innerWidth && box.bottom <= window.innerHeight
    const probed = [host, caption]
    probed.forEach((element) => element.style.setProperty('pointer-events', 'auto', 'important'))
    const hit = document.elementsFromPoint(box.left + box.width / 2, box.top + box.height / 2)[0] ?? null
    probed.forEach((element) => element.style.setProperty('pointer-events', 'none', 'important'))
    return {
      text: caption.textContent,
      open: caption.matches(':popover-open'),
      inViewport,
      painted: caption.checkVisibility({ opacityProperty: true, visibilityProperty: true }),
      topmost: hit === host,
    }
  }, ROOTS_KEY)
}

/** Collects the page's console warnings (the theme-host warnings among them) from now on. */
export function watchWarnings(page: Page): string[] {
  const warnings: string[] = []
  page.on('console', (message) => {
    if (message.type() === 'warning') warnings.push(message.text())
  })
  return warnings
}
