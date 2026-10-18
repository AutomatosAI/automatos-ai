/**
 * PRD-261 A-03 (PA-7) — builtin Play's display-face variable classes, for <html className>.
 *
 * The faces themselves stay in app/play-fonts.ts (next/font/local reads its options at build
 * time), with `preload: false`, so a file downloads only when a Play type style uses it and a
 * Studio or Classic page fetches none. The classes are host-generated (`__variable_<hash>`), the
 * same string on every render (PRD-260 M07), and never pass through a theme boot's addClasses.
 */
export { PLAY_FONT_VARIABLE_CLASSES } from '@/app/play-fonts'
