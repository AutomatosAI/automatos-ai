/**
 * Shared helpers for all Shepherd tours.
 */

/** Two-colour title: white text + gradient-text (orange accent) */
export function title(white: string, accent: string): string {
  return `<span style="color:#fff">${white}</span>&nbsp;<span class="gradient-text">${accent}</span>`
}

/** Wait for a DOM element to appear, with timeout fallback (resolves anyway to prevent hangs) */
export function waitForElement(selector: string, timeout = 1500): Promise<void> {
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

/**
 * Render a compact bullet list explaining what each top-level tab does.
 * Use inside a tour step to walk the user through a page's tab strip.
 *
 * Example:
 *   tabList([
 *     ['Documents', 'Upload files, connect cloud storage, run semantic search.'],
 *     ['Database', 'Connect SQL sources and query them in natural language.'],
 *   ])
 */
export function tabList(items: ReadonlyArray<readonly [string, string]>): string {
  const rows = items
    .map(
      ([label, desc]) =>
        `<li><strong class="text-gray-200">${label}</strong> <span class="text-gray-400">— ${desc}</span></li>`,
    )
    .join('')
  return `<ul class="shepherd-tab-list text-xs text-gray-300 mt-2 space-y-1 list-disc pl-4">${rows}</ul>`
}
