/**
 * PRD-260 P260-FX-22 (M21), PRD-261 A-14d — the mark on the h1 of a screen only Play has (Make it
 * yours, the style sheet). After 'Switch to Play' the host's note is replaced by the screen, and the
 * host's ThemeSlotPage moves the focus that dropped to <body> to the element it finds by
 * SCREEN_HEADING_SELECTOR (lib/theme-host/builtin/play-only-route.tsx). The SDK has no name for
 * that mark, so Play keeps its own copy of the attribute;
 * components/looks/__tests__/copied-helpers.test.tsx holds it to the host's selector.
 */

/** Spread on a Play-only screen's h1 (with tabIndex={-1}): where focus goes after Switch to Play. */
export const PLAY_ONLY_HEADING = { 'data-play-only-heading': true } as const
