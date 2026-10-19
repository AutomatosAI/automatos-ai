/**
 * PRD-261 A-03 (PA-7) — builtin Play's client module: it carries Play's stylesheet
 * (app/play-theme.css, every rule under html.play), so the stylesheet is in this module's chunk
 * and nowhere else. It is reached only through a dynamic import() (./clients.ts,
 * BUILTIN_THEME_CLIENTS, read by components/providers.tsx); the root layout never imports it, so
 * a Studio or Classic page requests none of Play's CSS. Rendered under Play on the server, next/dynamic writes the chunk's
 * stylesheet link into the first response, so Play's first paint is styled.
 *
 * Its default export is the component next/dynamic renders: it draws nothing.
 */
import '@/app/play-theme.css'

export default function BuiltinPlayClient(): null {
  return null
}
