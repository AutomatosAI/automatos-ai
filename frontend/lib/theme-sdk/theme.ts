/**
 * PRD-261 A-10a — SDK area `theme`: the SDK's version and defineTheme, the two names the A-01
 * spike shared. They stay in lib/theme-host/sdk.ts, which the loader reads the version from;
 * this area re-exports them so a theme reaches them through '@automatos/theme-sdk' as before.
 */
export { SDK_VERSION, defineTheme } from '@/lib/theme-host/sdk'
export type { ThemeDefinition, ThemeRegistrationHost } from '@/lib/theme-host/sdk'
