/**
 * PRD-261 — the words of the Themes storefront (the approved design, 30 Oct 2026: hero, looks,
 * strip, "Make your own" and the installed rows). The tab's own words (its name, Add to my
 * workspace, the badges) stay in ../listing-format.ts.
 */

export const STOREFRONT_TITLE = 'Make Automatos yours'
export const WEB_AND_PHONE = 'web + phone'

export function themeCount(count: number): string {
  return count === 1 ? '1 theme' : `${count} themes`
}

export const FEATURED_THEME = 'Featured theme'
export const FEATURED = 'Featured'
export const LOOKS_HEADING = 'Looks and screens'
export const DETAILS = 'Details'
/** An unpinned Marketplace install moves to a newly listed version (services/themes/auto_update.py). */
export const UPDATES_NOTE = 'Updates itself when a new version is listed · pin it to hold a version'
export const WEB_CAPTION = 'Web · Studio'
export const PHONE_CAPTION = 'Phone · Automatos Go'
export const PREVIOUS_THEME = 'Previous theme'
export const NEXT_THEME = 'Next theme'

export function showTheme(name: string): string {
  return `Show ${name}`
}

export function featuredPosition(position: number, count: number): string {
  return `${FEATURED} · ${position} of ${count}`
}

export function webScreenAlt(name: string, look: string): string {
  return `${name} on the web, ${look}`
}

export function phoneScreenAlt(name: string, look: string): string {
  return `${name} on Automatos Go, ${look}`
}

/** The label of the one view a listing has before its detail answers, or without previews. */
export const COVER_LOOK = 'Cover'
/** The label of a screenshot no preset claims (`otherPreviews`), numbered. */
export const OTHER_SCREEN = 'Screen'

export const ALL_THEMES = 'All themes'
export const SLIDE_FOR_MORE = 'Slide for more'
export const SCROLL_BACK = 'Scroll the themes back'
export const SCROLL_ON = 'Scroll the themes on'

export const MAKE_YOUR_OWN = 'Make your own theme'
export const MAKE_YOUR_OWN_BODY =
  'Design it with the Automatos Design Kit, for the web and for Automatos Go. Submit it here and, once reviewed, every workspace can install it.'
export const OPEN_DESIGN_KIT = 'Open the Design Kit'
export const DESIGN_KIT_URL = 'https://github.com/AutomatosAI/automatos-design-kit'
export const SUBMIT_PACKAGE = 'Submit a theme package for review'

export const EMPTY_TITLE = 'Be the first theme on the shelf'
export const EMPTY_BODY =
  'Design a theme with the Automatos Design Kit for the web and for Automatos Go, then submit it here: once reviewed, every workspace can install it.'

export const VIEW = 'View'

export function officialListed(version: string): string {
  return `Official ${version} is listed`
}

export function viewOfficial(name: string): string {
  return `View the Official ${name}`
}
