/**
 * Shared helpers for all Shepherd tours.
 */

/** Two-colour title: white text + gradient-text (orange accent) */
export function title(white: string, accent: string): string {
  return `<span style="color:#fff">${white}</span>&nbsp;<span class="gradient-text">${accent}</span>`
}

/** Wait for a DOM element to appear, with timeout fallback (resolves anyway to prevent hangs) */
export function waitForElement(selector: string, timeout = 5000): Promise<void> {
  return new Promise((resolve) => {
    if (typeof window === 'undefined') {
      resolve()
      return
    }

    if (document.querySelector(selector)) {
      resolve()
      return
    }

    let settled = false
    const settle = () => {
      if (settled) return
      settled = true
      observer.disconnect()
      clearTimeout(timer)
      resolve()
    }

    const observer = new MutationObserver(() => {
      if (document.querySelector(selector)) {
        settle()
      }
    })

    observer.observe(document.body, { childList: true, subtree: true })

    const timer = setTimeout(() => {
      if (!settled) {
        console.warn(`[tour] Element ${selector} not found within ${timeout}ms, skipping`)
        settle()
      }
    }, timeout)
  })
}

/** Standard Back/Next button pair */
export function backNextButtons(tour: any) {
  return [
    { text: 'Back', classes: 'shepherd-button-secondary', action: tour.back },
    { text: 'Next', action: tour.next },
  ]
}

/** Standard Next-only button */
export function nextButton(tour: any) {
  return [{ text: 'Next', action: tour.next }]
}

/** Standard finish button */
export function finishButton(tour: any) {
  return [{ text: 'Got it!', action: tour.complete }]
}

/** Step progress indicator (e.g. "Step 2 of 5") */
export function stepProgress(current: number, total: number): string {
  return `<div class="text-xs text-gray-500 mt-3">Step ${current} of ${total}</div>`
}
