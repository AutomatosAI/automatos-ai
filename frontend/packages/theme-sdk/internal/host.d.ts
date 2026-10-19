/**
 * PRD-261 A-10b — host types the SDK's declarations read but the SDK does not export by name
 * (frontend/lib/theme-host/registry-types.ts). Not re-exported by index.d.ts.
 */

/** The slots a theme may fill, each versioned with its props. */
export type SlotId = 'frame@1' | 'lookEditor@1' | 'styleSheet@1' | 'notFound@1' | 'asleep@1'
  | 'page-header.mascot@1' | 'page-header.art@1' | 'empty-state.mascot@1'
