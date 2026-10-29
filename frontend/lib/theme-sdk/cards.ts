/**
 * PRD-261 A-12a — SDK area `cards`: the host cards a theme places (PRD §4.3 item 7, §4.4).
 *
 * A theme places approval, task, withdraw and Mission cards and never draws its own versions of
 * them; its frame renders the host's HeaderUtilities. Each name is the host's own component behind
 * a small wrapper with the SDK's read-only props (host-cards/*), so the host can change its widgets
 * without changing the SDK, and a theme can import nothing of the host but this module.
 *
 * - approvals: ToolApproval, MissionApproval (protected surfaces, A-12c);
 * - chat reply cards: TaskCard, WithdrawOfferCard, MissionSuggestionCard, MissionCreatedCard;
 * - content: FilePreview, MarkdownView, MadeBy;
 * - chrome: HeaderUtilities, TrialExhaustedBanner, FirstRunNudge;
 * - the composer: AiLabelFence, once the host's AI label fence, now its children alone (PRD-264 EUAI-11a);
 * - CorePage: core's own page inside a theme's page.
 *
 * The names are the plan's (surface/cards.json); the types are declared again for themes in
 * packages/theme-sdk/areas/cards.d.ts.
 */
export { MissionApproval, ToolApproval } from './host-cards/approvals'
export type {
  ApprovalAnswer,
  ApprovalDecision,
  ApprovalViewData,
  CardOrigin,
  MissionApprovalData,
  MissionApprovalProps,
  MissionApprovalTask,
  OversightTier,
  ToolApprovalData,
  ToolApprovalProps,
} from './host-cards/approvals'

export { MissionCreatedCard, MissionSuggestionCard, TaskCard, WithdrawOfferCard } from './host-cards/chat-cards'
export type {
  MissionCreatedCardProps,
  MissionSuggestionCardProps,
  MissionSuggestionMessage,
  TaskCardData,
  TaskCardProps,
  WithdrawOffer,
  WithdrawOfferCardProps,
  WithdrawRequest,
} from './host-cards/chat-cards'

export { FilePreview, MadeBy, MarkdownView } from './host-cards/content'
export type { FilePreviewProps, FilePreviewType, MadeByProps, MarkdownViewProps } from './host-cards/content'

export { FirstRunNudge, HeaderUtilities, TrialExhaustedBanner } from './host-cards/chrome'
export type { TrialExhaustedBannerProps } from './host-cards/chrome'

export { AiLabelFence } from './host-cards/ai-label-fence'
export type { AiLabelFenceProps, AiLabelId } from './host-cards/ai-label-fence'

export { CorePage } from './host-cards/core-page'
export type { CorePageProps } from './host-cards/core-page'
