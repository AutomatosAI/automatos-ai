/**
 * PRD-261 A-13c — SDK area `assets` (frontend/lib/theme-sdk/assets.ts): the URL of a file inside the active theme's
 * package, served by the host's same-origin files route under the package's digest. The path is plain and relative
 * (lowercase letters, digits, '.', '_', '-' and '/'; no '..' and no '//'). Null when the path is refused, when no
 * installed theme is the style (a builtin theme's files are the app's own), when the host does not know the
 * package yet, and on the server.
 */

/** The URL of `path` (for example 'assets/mascot/wave.mp4') in the active theme's package, or null. */
export declare function assetUrl(path: string): string | null
