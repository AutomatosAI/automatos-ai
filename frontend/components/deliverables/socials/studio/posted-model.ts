/**
 * PRD-251C US-C408 — the Posted view's reading of a post that went out (pure): its numbers as
 * one line, each channel's receipt, and the filters' choices.
 */
import type { SocialNumberKey, SocialPostNumbers, SocialPostedReceipt } from '@/lib/socials-results-types'
import { channelLabel } from '../socials-status'

export const NOTHING_POSTED = 'Nothing has gone out yet with these filters. Posts show here once a channel publishes them.'
export const NO_NUMBERS = 'No numbers yet: a post is read a day after it goes out, and again after a week.'
const NUMBER_ORDER: ReadonlyArray<SocialNumberKey> = [
  'views', 'reach', 'likes', 'comments', 'shares', 'saves', 'replies', 'reposts', 'quotes', 'reactions',
]
const NUMBER_WORDS: Record<SocialNumberKey, [string, string]> = {
  views: ['view', 'views'], reach: ['reach', 'reach'], likes: ['like', 'likes'], comments: ['comment', 'comments'],
  shares: ['share', 'shares'], saves: ['save', 'saves'], replies: ['reply', 'replies'], reposts: ['repost', 'reposts'],
  quotes: ['quote', 'quotes'], reactions: ['reaction', 'reactions'],
}
const WEEK_READING = 7

export const FORMAT_FILTERS: ReadonlyArray<{ value: string; label: string }> = [
  { value: '', label: 'Any format' },
  { value: 'image', label: 'Image' },
  { value: 'carousel', label: 'Carousel' },
  { value: 'video', label: 'Video' },
  { value: 'fact_card', label: 'Fact card' },
  { value: 'text', label: 'Text only' },
]

export function countLabel(count: number, key: SocialNumberKey): string {
  const [one, many] = NUMBER_WORDS[key]
  return `${count.toLocaleString('en-GB')} ${count === 1 ? one : many}`
}

/** "940 views · 12 likes · 3 reposts (after a week)", or why there are none. */
export function numbersLine(numbers: SocialPostNumbers | null): string {
  if (!numbers) return NO_NUMBERS
  const when = numbers.reading >= WEEK_READING ? 'after a week' : 'after a day'
  const parts = NUMBER_ORDER.filter((key) => typeof numbers.numbers[key] === 'number')
    .map((key) => countLabel(numbers.numbers[key] as number, key))
  return parts.length ? `${parts.join(' · ')} (${when})` : `The channels gave no numbers (${when}).`
}

/** "Instagram story", "X": where a receipt's post went out. */
export function receiptLabel(receipt: Pick<SocialPostedReceipt, 'toolkit' | 'post_kind'>): string {
  return receipt.post_kind === 'story' ? `${channelLabel(receipt.toolkit)} story` : channelLabel(receipt.toolkit)
}

/** "Fri 23 Oct, 09:00" in the viewer's zone; "" when unknown. */
export function wentOutLabel(iso: string | null): string {
  if (!iso) return ''
  const moment = new Date(iso)
  if (Number.isNaN(moment.getTime())) return ''
  return moment.toLocaleString('en-GB', { weekday: 'short', day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' })
}
