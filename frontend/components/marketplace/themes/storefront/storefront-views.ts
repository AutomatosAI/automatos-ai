/**
 * PRD-261 — the storefront hero's "looks and screens": what each chip shows on the web (the browser
 * frame) and on Automatos Go (the phone frame), from a listing's detail (listings.py
 * previews_by_preset, plus the storefront contract's phone screenshots).
 *
 * - One view per preset and screen: the screens are the preset's web screens, then any phone screen
 *   it has no web screen for. A view's web screen is the preset's screenshot of that screen, else the
 *   preset's first; its phone screen likewise, else the listing's `phoneCover`, else none (the phone
 *   frame hides). Every screenshot of a preset is reachable from a chip.
 * - The previews no preset claims (`otherPreviews`, `otherPhonePreviews`) follow, paired in order.
 * - The view showing the card's cover comes first, so the hero keeps its image when the detail
 *   answers. Before it answers, or with no previews at all, the one view is the cover.
 *
 * Every URL goes through `themeFileUrl`, so nothing here points at another origin.
 */
import type { ThemeListing, ThemeListingDetail, ThemePresetPreviews, ThemePreviewShot } from '@/lib/api-client'

import { themeFileUrl } from '../listing-format'
import { COVER_LOOK, OTHER_SCREEN } from './copy'

export interface StorefrontView {
  readonly key: string
  readonly label: string
  /** The web screenshot (a files-route URL), or null. */
  readonly web: string | null
  /** The Automatos Go screenshot (a files-route URL), or null: the phone frame hides. */
  readonly phone: string | null
}

const COVER_KEY = 'cover'
const SCREEN_SEPARATOR = ' · '
const WORD_BREAKS = /[-_]+/g

/** A preset id as a chip's label: `night-city` reads "Night city". */
export function lookLabel(id: string): string {
  const words = id.replace(WORD_BREAKS, ' ').trim()
  return words ? `${words[0].toUpperCase()}${words.slice(1)}` : id
}

function screensOf(web: readonly ThemePreviewShot[], phone: readonly ThemePreviewShot[]): (string | null)[] {
  const screens = [...web, ...phone].map((shot) => shot.screen)
  return screens.filter((screen, index) => screens.indexOf(screen) === index)
}

function shotFor(shots: readonly ThemePreviewShot[], screen: string | null): string | null {
  const shot = shots.find((candidate) => candidate.screen === screen) ?? shots[0]
  return themeFileUrl(shot?.url)
}

function presetViews(preset: ThemePresetPreviews): StorefrontView[] {
  const web = preset.previews
  const phone = preset.phonePreviews ?? []
  const look = lookLabel(preset.id)
  return screensOf(web, phone).map((screen) => ({
    key: `${preset.id}:${screen ?? ''}`,
    label: screen ? `${look}${SCREEN_SEPARATOR}${screen}` : look,
    web: shotFor(web, screen),
    phone: shotFor(phone, screen),
  }))
}

function otherViews(detail: ThemeListingDetail): StorefrontView[] {
  const web = detail.otherPreviews
  const phone = detail.otherPhonePreviews ?? []
  return Array.from({ length: Math.max(web.length, phone.length) }, (_, index) => ({
    key: `other:${index}`,
    label: `${OTHER_SCREEN} ${index + 1}`,
    web: themeFileUrl(web[index] ?? web[0]),
    phone: themeFileUrl(phone[index] ?? phone[0]),
  }))
}

function isDetail(listing: ThemeListing | ThemeListingDetail): listing is ThemeListingDetail {
  return 'presets' in listing && Array.isArray(listing.presets)
}

function coverFirst(views: readonly StorefrontView[], cover: string | null): StorefrontView[] {
  const at = cover === null ? -1 : views.findIndex((view) => view.web === cover)
  return at > 0 ? [views[at], ...views.slice(0, at), ...views.slice(at + 1)] : [...views]
}

/** The hero's views of a listing: from its detail once it answers, else its cover alone. */
export function storefrontViews(listing: ThemeListing | ThemeListingDetail): StorefrontView[] {
  const cover = themeFileUrl(listing.cover)
  const phoneCover = themeFileUrl(listing.phoneCover)
  const views = isDetail(listing) ? [...listing.presets.flatMap(presetViews), ...otherViews(listing)] : []
  const usable = views.filter((view) => view.web !== null || view.phone !== null)
  if (usable.length === 0) return [{ key: COVER_KEY, label: COVER_LOOK, web: cover, phone: phoneCover }]
  return coverFirst(usable, cover).map((view) => ({ ...view, phone: view.phone ?? phoneCover }))
}

/** A look index kept inside the views, so a detail with fewer views never points past them. */
export function clampLook(views: readonly StorefrontView[], look: number): number {
  return Math.min(Math.max(look, 0), views.length - 1)
}
