/**
 * PRD-261 A-03 — first paint from data. The root layout resolves the style's boot (its <html>
 * classes, its tone, the look it paints before first paint, its module integrity) and writes it
 * into the page as one JSON element (components/theme-host/theme-boot.tsx); THEME_BOOT_SCRIPT,
 * one constant for every style, reads that element and applies it. No script is ever built from
 * data, and the boot itself passes sanitizeThemeBoot first (boot-sanitize.ts).
 *
 * React never owns the style classes: <html className> carries only the font variables, the
 * same on every render (PRD-260 M07), so a server re-render (router.refresh after a client-side
 * style switch) changes only the JSON element's text, which no script reads again, and the tone
 * class next-themes put on <html> survives.
 *
 * Every style gets correct first-paint classes: Classic none, Studio `studio`, a builtin theme
 * its base's class and its own, and an installed theme (`theme:install:<uuid>`) its base's —
 * Studio, DEFAULT_THEME_BASE — until A-08c's resolver returns its own boot. An installed theme's
 * boot names its own look keys and guard (W-A6b, gap G2); when its look sets the tone, its tone
 * comes from the generic look runtime's tone cookie (A-19d2: `automatos-theme-tone`), as builtin
 * Play's comes from its own.
 */
import { STYLE_HTML_CLASS_NAMES, styleHtmlClasses } from '@/lib/ui-style'

import { parseThemeTone, sanitizeThemeBoot } from './boot-sanitize'
import type { BuiltinThemeBootData, CookieReader, ThemeBoot, ThemeBootResolver, ThemeTone } from './boot-types'
import { BUILTIN_PLAY_BOOT } from './builtin/play-boot'
import { BUILTIN_PLAY_REF } from './builtin/play-entry'
import { themeLookRulesOf } from './looks/spec'
import { parseThemeToneCookie, THEME_TONE_COOKIE } from './looks/tone-cookie'
import { findThemeEntry, isInstalledThemeStyle, themeRefOf } from './registry'
import type { ThemeTonePolicy, UiStyle } from './registry-types'

export type {
  BuiltinThemeBootData,
  CookieReader,
  InstalledThemeEntryData,
  PrepaintKnobTable,
  PrepaintNumericRule,
  PrepaintPair,
  PrepaintPairTable,
  PrepaintTable,
  PrepaintVars,
  ThemeBoot,
  ThemeBootResolver,
  ThemeLookGuard,
  ThemeLookStorage,
  ThemeTone,
} from './boot-types'

/** The id of the JSON element THEME_BOOT_SCRIPT reads. */
export const THEME_BOOT_ELEMENT_ID = 'automatos-theme-boot'

/** The first-paint data of each theme compiled into the app, by its ref. */
const BUILTIN_THEME_BOOTS: ReadonlyMap<string, BuiltinThemeBootData> = new Map([[BUILTIN_PLAY_REF, BUILTIN_PLAY_BOOT]])

/** No theme, or a theme the registry does not know yet: the user's own tone. */
const USER_TONE_POLICY: ThemeTonePolicy = 'user'

export { parseThemeTone }

/** The tone a policy fixes on its own: 'light' and 'dark' do, 'user' and 'from-look' do not. */
function fixedTone(policy: ThemeTonePolicy): ThemeTone | null {
  return parseThemeTone(policy)
}

/** The boot of a style with nothing to paint: its classes, the others removed, and its tone policy. */
export function baseThemeBoot(style: UiStyle): ThemeBoot {
  const addClasses = styleHtmlClasses(style)
  const policy = findThemeEntry(style)?.tone ?? USER_TONE_POLICY
  return sanitizeThemeBoot({
    addClasses,
    removeClasses: STYLE_HTML_CLASS_NAMES.filter((name) => !addClasses.includes(name)),
    tone: { policy, value: fixedTone(policy) },
    prepaint: null,
    lookStorage: null,
    lookGuard: null,
    integrity: null,
    entry: null,
  })
}

/**
 * The boot of Classic, Studio or a builtin theme (with its look and its tone cookie), and the
 * base boot of an installed theme. Synchronous: everything it reads is compiled in or a cookie.
 */
export function builtinThemeBoot(style: UiStyle, readCookie: CookieReader): ThemeBoot {
  const base = baseThemeBoot(style)
  const ref = themeRefOf(style)
  const data = ref === null ? undefined : BUILTIN_THEME_BOOTS.get(ref)
  if (data === undefined) return base
  const cookie = data.toneCookie
  const value = base.tone.policy === 'from-look' && cookie !== null ? parseThemeTone(readCookie(cookie)) : base.tone.value
  return sanitizeThemeBoot({
    ...base,
    tone: { policy: base.tone.policy, value },
    prepaint: data.prepaint,
    lookStorage: data.lookStorage,
    lookGuard: data.lookGuard,
  })
}

const LOG_PREFIX = '[theme-boot]'

/**
 * Whether the generic look runtime (lib/theme-host/looks/provider.tsx) keeps this installed boot's
 * look tone in THEME_TONE_COOKIE: a running theme (its entry) whose look sets the tone, whose boot
 * names where its look is kept, has looks to run (its table, guard and presets) and a tone rule in
 * its table. Exactly then the runtime writes the cookie; any other theme would start on another
 * theme's last tone and keep it, since nothing would hand it a tone after hydration.
 */
function keepsLookTone(boot: ThemeBoot): boolean {
  if (boot.tone.policy !== 'from-look' || boot.lookStorage === null || boot.entry === null) return false
  return boot.prepaint?.tone !== undefined && themeLookRulesOf(boot) !== null
}

/**
 * PRD-261 W-A6b (gap G2), A-19d2: an installed theme whose look sets the tone starts on the tone
 * its look last wrote to the generic look runtime's tone cookie, so the first paint is already that
 * tone. The cookie is read per request, after the backend's boot (which the boot cache may hold) is
 * answered, and strictly: only this theme's own value (its tone and its id's mark, tone-cookie.ts)
 * is a tone, so a theme switched to from another installed one never starts on that one's tone.
 * Builtin Play's own cookie is never read for an installed theme, Play's package included.
 */
function withLookTone(boot: ThemeBoot, readCookie: CookieReader): ThemeBoot {
  if (boot.entry === null || !keepsLookTone(boot)) return boot
  const value = parseThemeToneCookie(readCookie(THEME_TONE_COOKIE), boot.entry.themeId)
  return { ...boot, tone: { policy: boot.tone.policy, value } }
}

/**
 * The resolver the root layout calls. `installed` (A-08c) answers for `theme:install:*`; its
 * boot passes the same gate, and when it fails the page gets the base boot (Studio's classes)
 * and the client host takes over. A 'from-look' installed boot with looks and a tone rule takes
 * its tone from the generic look runtime's tone cookie (withLookTone).
 */
export function createThemeBootResolver(readCookie: CookieReader, installed?: ThemeBootResolver): ThemeBootResolver {
  return async (style) => {
    if (installed === undefined || !isInstalledThemeStyle(style)) return builtinThemeBoot(style, readCookie)
    try {
      return withLookTone(sanitizeThemeBoot(await installed(style)), readCookie)
    } catch (error) {
      console.error(LOG_PREFIX, 'installed theme boot failed; the base boot paints instead', error)
      return baseThemeBoot(style)
    }
  }
}

/**
 * The one constant boot script (plan §2.2). It reads the JSON element, puts the boot's classes
 * on <html> (removing the other style classes and leaving every other class alone), then, when
 * the boot carries a prepaint table, paints the stored active look, or the default look when
 * nothing valid is stored or storage is blocked. It applies only with classList and
 * style.setProperty, and re-checks every class against HTML_CLASS_PATTERN and every property
 * name against CSS_VAR_NAME_PATTERN (both written here as literals; boot.test.ts holds them
 * equal). A literal: nothing is ever built into it.
 */
export const THEME_BOOT_SCRIPT = `(function(){
var d=document.documentElement,c=d.classList,own=Object.prototype.hasOwnProperty,b=null;
try{var e=document.getElementById("automatos-theme-boot");b=e?JSON.parse(e.textContent||"null"):null;}catch(x){b=null;}
if(b===null||typeof b!=="object"){return;}
var C=/^[a-z][a-z0-9-]{2,31}$/,V=/^--[a-z0-9-]{1,64}$/;
function arr(a){return Array.isArray(a)?a:[];}
function obj(o){return o!==null&&typeof o==="object";}
function has(o,k){return obj(o)&&typeof k==="string"&&own.call(o,k);}
function classes(a,m){a=arr(a);for(var i=0;i<a.length;i++){if(typeof a[i]==="string"&&C.test(a[i])){c[m](a[i]);}}}
classes(b.removeClasses,"remove");classes(b.addClasses,"add");
var P=b.prepaint;if(!obj(P)){return;}
var K=obj(P.knobs)?P.knobs:{},Q=arr(P.pairs),N=arr(P.numeric);
function num(r,l){if(!obj(r)){return false;}var v=l[r.knob];if(typeof v!=="number"||!(v>=r.min&&v<=r.max)||(r.integer&&v%1!==0)){return false;}if(!obj(r.color)||!has(r.color.rgb,l[r.color.knob])){return false;}return !r.capTable||(has(r.caps,l[r.capTable[0]])&&has(r.caps[l[r.capTable[0]]],l[r.capTable[1]]));}
function fits(l){if(!obj(l)){return false;}for(var k in K){if(own.call(K,k)&&!has(K[k],l[k])){return false;}}for(var i=0;i<Q.length;i++){if(!obj(Q[i])||!has(Q[i].table,l[Q[i].a])||!has(Q[i].table[l[Q[i].a]],l[Q[i].b])){return false;}}for(var j=0;j<N.length;j++){if(!num(N[j],l)){return false;}}return true;}
function guarded(l){var g=b.lookGuard;return !obj(g)||(l.schema===g.schema&&l.interface===g.iface);}
function put(v){if(!obj(v)){return;}for(var k in v){if(own.call(v,k)&&V.test(k)&&typeof v[k]==="string"){d.style.setProperty(k,v[k]);}}}
var l=null,S=b.lookStorage;
if(obj(S)){try{var s=window.localStorage,p=s.getItem(S.pointerKey);l=JSON.parse(s.getItem(p?S.scopedPrefix+p:S.fallbackKey)||"null");}catch(x){l=null;}}
if(!fits(l)||!guarded(l)){l=P.defaults;if(!fits(l)){return;}}
for(var k in K){if(own.call(K,k)){put(K[k][l[k]]);}}
for(var i=0;i<Q.length;i++){put(Q[i].table[l[Q[i].a]][l[Q[i].b]]);}
for(var j=0;j<N.length;j++){var r=N[j];if(r.template!=="rgba-alpha"||typeof r.var!=="string"||!V.test(r.var)){continue;}var a=l[r.knob]/r.scale;if(r.capTable){a=Math.min(a,r.caps[l[r.capTable[0]]][l[r.capTable[1]]]);}d.style.setProperty(r.var,"rgba("+r.color.rgb[l[r.color.knob]]+","+a+")");}
})();`
