/**
 * Tour ↔ Modal tab bridge using CustomEvents.
 *
 * Flow:
 *  1. Tour dispatches `tour:set-modal-tab` with target tab number
 *  2. Modal's useTourTabBridge hook calls setStep(), then confirms via `tour:modal-tab-ready`
 *  3. switchTabAndWaitForElement() waits for confirmation + element presence, then resolves
 */

export const TOUR_EVENTS = {
  SET_MODAL_TAB: 'tour:set-modal-tab',
  MODAL_TAB_READY: 'tour:modal-tab-ready',
} as const

/** Dispatch a request for the modal to switch to a specific tab */
export function requestModalTab(step: number) {
  window.dispatchEvent(
    new CustomEvent(TOUR_EVENTS.SET_MODAL_TAB, { detail: { step } })
  )
}

/**
 * Switch the modal to `targetTab`, then wait for `selector` to appear in the DOM.
 * Returns a Promise suitable for Shepherd's `beforeShowPromise`.
 */
export function switchTabAndWaitForElement(
  targetTab: number,
  selector: string,
  timeout = 5000
): Promise<void> {
  return new Promise((resolve, reject) => {
    const el = document.querySelector(selector)
    // If element is already visible, resolve immediately
    if (el) {
      requestModalTab(targetTab)
      // Still give a frame for React to settle
      requestAnimationFrame(() => resolve())
      return
    }

    let settled = false
    const settle = () => {
      if (settled) return
      settled = true
      observer.disconnect()
      clearTimeout(timer)
      // 150ms settle delay for framer-motion animations
      setTimeout(resolve, 150)
    }

    // Listen for modal confirmation that tab switched
    const onReady = () => {
      window.removeEventListener(TOUR_EVENTS.MODAL_TAB_READY, onReady)
      // After tab switch confirmed, check if element exists or wait for it
      if (document.querySelector(selector)) {
        settle()
        return
      }
      // MutationObserver fallback — element may render a frame later
    }
    window.addEventListener(TOUR_EVENTS.MODAL_TAB_READY, onReady)

    // MutationObserver to catch the element appearing
    const observer = new MutationObserver(() => {
      if (document.querySelector(selector)) {
        settle()
      }
    })
    observer.observe(document.body, { childList: true, subtree: true })

    // Timeout
    const timer = setTimeout(() => {
      if (!settled) {
        settled = true
        observer.disconnect()
        window.removeEventListener(TOUR_EVENTS.MODAL_TAB_READY, onReady)
        // Resolve anyway so the tour doesn't hang — step will just miss its target
        console.warn(`[tour-bridge] Timed out waiting for ${selector}`)
        resolve()
      }
    }, timeout)

    // Fire the tab switch request
    requestModalTab(targetTab)
  })
}
