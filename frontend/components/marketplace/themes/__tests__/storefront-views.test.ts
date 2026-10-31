/**
 * PRD-261 — the storefront hero's looks and screens (storefront-views.ts): one view per preset and
 * screen, the phone screenshot matched to it (else the preset's first, else the listing's
 * phoneCover, else none), the cover's view first, the unclaimed previews after, and nothing from
 * another origin.
 */
import { describe, expect, it } from 'vitest'

import { COVER_LOOK, OTHER_SCREEN } from '../storefront/copy'
import { clampLook, lookLabel, storefrontViews } from '../storefront/storefront-views'
import { FILES, listing, listingDetail } from './listing-fixtures'

const WEB = `${FILES}/assets/previews`
const PHONE = `${FILES}/assets/previews/phone`
const ELSEWHERE = 'https://evil.example/shot.png'

function hudDetail(overrides: Parameters<typeof listingDetail>[0] = {}) {
  return listingDetail({
    cover: `${WEB}/arc.png`,
    phoneCover: `${PHONE}/arc.png`,
    presets: [
      {
        id: 'arc',
        default: true,
        previews: [{ screen: null, url: `${WEB}/arc.png` }, { screen: 'chat', url: `${WEB}/arc-chat.png` }],
        phonePreviews: [{ screen: null, url: `${PHONE}/arc.png` }, { screen: 'home', url: `${PHONE}/arc-home.png` }],
      },
      { id: 'night-city', default: false, previews: [{ screen: null, url: `${WEB}/night-city.png` }] },
    ],
    otherPreviews: [],
    ...overrides,
  })
}

describe('storefrontViews', () => {
  it('before the detail answers, is the card’s cover and phone cover alone', () => {
    const views = storefrontViews(listing({ cover: `${WEB}/arc.png`, phoneCover: `${PHONE}/arc.png` }))
    expect(views).toEqual([{ key: 'cover', label: COVER_LOOK, web: `${WEB}/arc.png`, phone: `${PHONE}/arc.png` }])
  })

  it('is one view per preset and screen: web screens, then the phone screens with no web screen', () => {
    const views = storefrontViews(hudDetail())
    expect(views.map((view) => view.label)).toEqual(['Arc', 'Arc · chat', 'Arc · home', 'Night city'])
    expect(views.map((view) => view.web)).toEqual([
      `${WEB}/arc.png`, `${WEB}/arc-chat.png`, `${WEB}/arc.png`, `${WEB}/night-city.png`,
    ])
  })

  it('matches a phone screenshot by screen, else the preset’s first, else the listing’s phoneCover', () => {
    const views = storefrontViews(hudDetail({ phoneCover: `${PHONE}/cover.png` }))
    expect(views.map((view) => view.phone)).toEqual([
      `${PHONE}/arc.png`, // the preset's own
      `${PHONE}/arc.png`, // no phone "chat": the preset's first
      `${PHONE}/arc-home.png`,
      `${PHONE}/cover.png`, // no phone screenshot in the preset: the listing's phoneCover
    ])
  })

  it('has no phone screen when the listing has no phone screenshots at all', () => {
    const views = storefrontViews(listingDetail())
    expect(views.length).toBeGreaterThan(0)
    expect(views.every((view) => view.phone === null)).toBe(true)
  })

  it('puts the view of the card’s cover first, so the hero keeps its image when the detail answers', () => {
    const views = storefrontViews(hudDetail({ cover: `${WEB}/night-city.png` }))
    expect(views[0].web).toBe(`${WEB}/night-city.png`)
    expect(views.map((view) => view.label)).toEqual(['Night city', 'Arc', 'Arc · chat', 'Arc · home'])
  })

  it('adds the previews no preset claims, web and phone paired in order', () => {
    const views = storefrontViews(
      hudDetail({
        presets: [],
        otherPreviews: [`${WEB}/one.png`, `${WEB}/two.png`],
        otherPhonePreviews: [`${PHONE}/one.png`],
      }),
    )
    expect(views.map((view) => [view.label, view.web, view.phone])).toEqual([
      [`${OTHER_SCREEN} 1`, `${WEB}/one.png`, `${PHONE}/one.png`],
      [`${OTHER_SCREEN} 2`, `${WEB}/two.png`, `${PHONE}/one.png`],
    ])
  })

  it('never keeps a URL from another origin, and falls back to the cover when nothing is left', () => {
    const views = storefrontViews(
      hudDetail({
        cover: ELSEWHERE,
        phoneCover: ELSEWHERE,
        presets: [{ id: 'arc', default: true, previews: [{ screen: null, url: ELSEWHERE }], phonePreviews: [{ screen: null, url: ELSEWHERE }] }],
        otherPreviews: [ELSEWHERE],
      }),
    )
    expect(views).toEqual([{ key: 'cover', label: COVER_LOOK, web: null, phone: null }])
  })
})

describe('lookLabel and clampLook', () => {
  it('reads a preset id as words', () => {
    expect(lookLabel('night-city')).toBe('Night city')
    expect(lookLabel('arc')).toBe('Arc')
    expect(lookLabel('neon_v2')).toBe('Neon v2')
  })

  it('keeps a look inside the views', () => {
    const views = storefrontViews(hudDetail())
    expect(clampLook(views, 9)).toBe(views.length - 1)
    expect(clampLook(views, -1)).toBe(0)
    expect(clampLook(views, 1)).toBe(1)
  })
})
