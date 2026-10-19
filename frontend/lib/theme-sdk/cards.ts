/**
 * PRD-261 A-10a — SDK area `cards`: the host cards a theme places (PRD §4.3 item 7). Stubbed here; A-12a fills the bodies.
 * Until then every name throws "not in this SDK build" (internal/stub.ts), so a theme built
 * against a later SDK fails loudly. The names are the plan's (surface/cards.json): the
 * area's lane replaces bodies, never names, unless the A-10c surface map changes first.
 */
import { notInThisBuild } from './internal/stub'

const AREA = 'cards'

export const ToolApproval = notInThisBuild(AREA, 'ToolApproval')
export const MissionApproval = notInThisBuild(AREA, 'MissionApproval')
export const TaskCard = notInThisBuild(AREA, 'TaskCard')
export const WithdrawOfferCard = notInThisBuild(AREA, 'WithdrawOfferCard')
export const MissionSuggestionCard = notInThisBuild(AREA, 'MissionSuggestionCard')
export const MissionCreatedCard = notInThisBuild(AREA, 'MissionCreatedCard')
export const FilePreview = notInThisBuild(AREA, 'FilePreview')
export const MarkdownView = notInThisBuild(AREA, 'MarkdownView')
export const MadeBy = notInThisBuild(AREA, 'MadeBy')
export const TrialExhaustedBanner = notInThisBuild(AREA, 'TrialExhaustedBanner')
export const FirstRunNudge = notInThisBuild(AREA, 'FirstRunNudge')
export const HeaderUtilities = notInThisBuild(AREA, 'HeaderUtilities')
export const CorePage = notInThisBuild(AREA, 'CorePage')
