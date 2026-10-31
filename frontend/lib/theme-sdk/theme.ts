/**
 * PRD-261 A-10a — SDK area `theme`: the SDK's version and defineTheme, the two names the A-01
 * spike shared. They stay in lib/theme-host/sdk.ts, which the loader reads the version from;
 * this area re-exports them so a theme reaches them through '@automatos/theme-sdk' as before.
 *
 * PRD-268 US-C21e (SDK 1.3.0): DISPLAY_STATE_LABELS, §6.11.1's glyph and word for each row state
 * (lib/proactive/display-state.ts), so a theme that draws a row's status itself says it as the host does.
 */
export { SDK_VERSION, defineTheme } from '@/lib/theme-host/sdk'
export type { ThemeDefinition, ThemeRegistrationHost } from '@/lib/theme-host/sdk'
export { DISPLAY_STATE_LABELS } from '@/lib/proactive/display-state'
export type { StateLabel as DisplayStateLabel, StatusRole, StatusToken } from '@/lib/proactive/display-state'
