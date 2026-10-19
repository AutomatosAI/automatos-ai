/**
 * PRD-261 A-10a — a planned SDK name whose area has not landed in this build.
 *
 * Every name a later lane fills (chat, deliverables, cards, looks, assets, pages/*) exists from
 * A-10a on, so the surface, the shim and a theme's imports never change shape when the area
 * lands; only the body does. Until then calling it throws, and the message names the name and
 * the area, so a theme built against a newer SDK fails loudly instead of rendering nothing.
 */

/** The words every stub's error message carries. */
export const NOT_IN_THIS_BUILD = 'not in this SDK build'

/** What a stub is: callable with anything, and it always throws. */
export type SdkStub = (...args: readonly unknown[]) => never

/** A stub for `name` in `area`: a function named `name` that throws when called. */
export function notInThisBuild(area: string, name: string): SdkStub {
  const stub: SdkStub = () => {
    throw new Error(`Automatos theme SDK: ${name} (${area}) is ${NOT_IN_THIS_BUILD}.`)
  }
  Object.defineProperty(stub, 'name', { value: name })
  return stub
}
