import type { Locator } from '@playwright/test'

// PRD-261 A-17b: what the page actually draws, read from computed styles, so a Play spec checks
// installed Play (the theme kit's package) by the colours on screen, never by class names (a CSS
// module's class is hashed, and the package's stylesheet is built by the kit).

/**
 * The colour `value` (any CSS colour, such as 'var(--pl-color-accent-text)') resolves to inside
 * `scope`, written the way getComputedStyle writes colours ('rgb(147, 68, 27)'): a probe element
 * is added there, read and removed.
 */
export function resolvedColour(scope: Locator, value: string): Promise<string> {
  return scope.evaluate((element, colour) => {
    const probe = document.createElement('span')
    probe.style.color = colour
    element.appendChild(probe)
    const resolved = getComputedStyle(probe).color
    probe.remove()
    return resolved
  }, value)
}

/** One computed property of the element. */
export function computedStyleOf(target: Locator, property: string): Promise<string> {
  return target.evaluate((element, name) => getComputedStyle(element).getPropertyValue(name), property)
}
