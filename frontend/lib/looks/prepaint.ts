import { PLAY_HTML_CLASS } from '@/lib/ui-style'

import {
  ACCENTS,
  ACCENT_KEYS,
  GROUND_KEYS,
  LIGHT_KEYS,
  SHELL_KEYS,
  TYPE_STYLE_KEYS,
} from './catalogue'
import {
  GLOW_SCALE,
  GLOW_VAR,
  accentTextVars,
  accentVars,
  fontVars,
  groundVars,
  lightVars,
  rgbList,
  safeGlowAlpha,
  shellVars,
  type CssVars,
} from './css'
import { DEFAULT_LOOK } from './presets'
import { scopedKey } from './scope'
import { ACTIVE_LOOK_KEY, LOOK_SCOPE_KEY } from './storage'
import { studioAccentTokens, studioGroundTokens } from './studio-tokens'
import { GLOW_MAX, GLOW_MIN, LOOK_INTERFACE, LOOK_SCHEMA } from './types'

/**
 * PRD-260 D4, K3 — the inline <head> script that puts the stored look on
 * <html> before first paint, under Play only (app/layout.tsx renders it only
 * when the style is play, and the script itself does nothing without the
 * 'play' class). It sets Play's --pl-* variables AND Studio's tokens, so a
 * Studio page under Play paints in the look's colours from the first frame.
 *
 * Its body is a constant, built here at module load from the catalogue
 * through the same builders as lookToCssVars() and lookToStudioTokens(). The
 * only runtime input is the stored active look, used only after every field
 * is checked against the catalogue's own keys; anything else, a corrupt or
 * blocked store included, gets the default look (the LookProvider reads the
 * same store after hydration and logs why). Looks are kept per account (F5)
 * and this runs before anyone is signed in, so it reads the active look of
 * the scope LOOK_SCOPE_KEY names (the account under Play on the last load);
 * with no pointer yet, the unscoped key a browser kept before F5. The pointer
 * only names a localStorage key; it is never painted. It checks the fields it paints
 * with, plus the schema; a look that validateLook() refuses only for its
 * name or id paints here and gives way to the default once storage is read.
 * Never put user text into this string; it writes custom properties only,
 * never markup. The glow is painted as glowVars() paints it: never past the
 * ground and accent's safe glow (table c, from safeGlowAlpha()).
 *
 * The tone is not set here: next-themes' own script runs later in <body> and
 * owns the light/dark class, so the tone reaches the first paint through its
 * cookie and ThemeProvider's forcedTheme (lib/looks/tone.ts).
 */

function table<K extends string, V>(keys: readonly K[], vars: (key: K) => V): Readonly<Record<string, V>> {
  return Object.fromEntries(keys.map((key) => [key, vars(key)]))
}

/** Every catalogue entry's variables, keyed the way the script reads them. */
export function prepaintTable() {
  const { ground, accent, font, shell, light, glow } = DEFAULT_LOOK
  return {
    g: table(GROUND_KEYS, (key) => ({ ...groundVars(key), ...studioGroundTokens(key) })),
    a: table(ACCENT_KEYS, accentVars),
    x: Object.fromEntries(
      GROUND_KEYS.map((key) => [key, table(ACCENT_KEYS, (other) => ({ ...accentTextVars(key, other), ...studioAccentTokens(key, other) }))]),
    ),
    f: table(TYPE_STYLE_KEYS, fontVars),
    s: table(SHELL_KEYS, shellVars),
    l: table(LIGHT_KEYS, lightVars),
    r: Object.fromEntries(ACCENT_KEYS.map((key) => [key, rgbList(ACCENTS[key].hex)])),
    c: table(GROUND_KEYS, (key) => table(ACCENT_KEYS, (other) => safeGlowAlpha(key, other))),
    d: { ground, accent, font, shell, light, glow },
  }
}

/** JSON that can sit inside a <script> element: no '</script>', no '<!--', no line separators. */
export function scriptSafeJson(value: unknown): string {
  return JSON.stringify(value)
    .replace(/</g, '\\u003c')
    .replace(/\u2028/g, '\\u2028')
    .replace(/\u2029/g, '\\u2029')
}

const SCHEMA = scriptSafeJson(LOOK_SCHEMA)
const INTERFACE = scriptSafeJson(LOOK_INTERFACE)

export const LOOK_PREPAINT_SCRIPT = `(function(){
var d=document.documentElement;
if(!d.classList.contains(${scriptSafeJson(PLAY_HTML_CLASS)})){return;}
var T=${scriptSafeJson(prepaintTable())},own=Object.prototype.hasOwnProperty;
function has(o,k){return typeof k==="string"&&own.call(o,k);}
function valid(l){return l!==null&&typeof l==="object"&&l.schema===${SCHEMA}&&l.interface===${INTERFACE}&&has(T.g,l.ground)&&has(T.a,l.accent)&&has(T.f,l.font)&&has(T.s,l.shell)&&has(T.l,l.light)&&typeof l.glow==="number"&&l.glow%1===0&&l.glow>=${GLOW_MIN}&&l.glow<=${GLOW_MAX};}
function put(v){for(var k in v){if(own.call(v,k)){d.style.setProperty(k,v[k]);}}}
var l=null;
try{var s=window.localStorage,p=s.getItem(${scriptSafeJson(LOOK_SCOPE_KEY)});l=JSON.parse(s.getItem(p?${scriptSafeJson(scopedKey(ACTIVE_LOOK_KEY, ''))}+p:${scriptSafeJson(ACTIVE_LOOK_KEY)})||"null");}catch(e){l=null;}
if(!valid(l)){l=T.d;}
put(T.g[l.ground]);put(T.a[l.accent]);put(T.x[l.ground][l.accent]);put(T.f[l.font]);put(T.s[l.shell]);put(T.l[l.light]);
d.style.setProperty(${scriptSafeJson(GLOW_VAR)},"rgba("+T.r[l.accent]+","+Math.min(l.glow/${GLOW_SCALE},T.c[l.ground][l.accent])+")");
})();`
