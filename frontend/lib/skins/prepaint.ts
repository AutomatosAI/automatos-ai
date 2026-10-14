// PRD-259 §4 — the inline <head> script that applies the active skin before first paint (the
// pattern of Play's lib/looks/prepaint.ts, re-implemented). Its body is a constant built here
// at module load from constants only: the cache key, the element id, the id pattern and the
// stylesheet grammar. The only runtime inputs are <html data-skin> (set by the server) and the
// cached stylesheet, and the cache is used only when it is for that exact skin and matches
// SKIN_CSS_GRAMMAR. It is set as textContent, never innerHTML. Anything else does nothing:
// the base style shows and the provider repairs after hydration.

import { SKIN_CSS_CACHE_KEY } from './cache'
import { SKIN_CSS_GRAMMAR, SKIN_DATA_ATTRIBUTE } from './css'
import { SKIN_ID_PATTERN } from './schema'

/** The one <style> element a skin's stylesheet lives in. */
export const SKIN_STYLE_ELEMENT_ID = 'automatos-skin'

/** A JS literal that can sit inside a <script> element: no "</script>", no line separators. */
function scriptLiteral(value: string): string {
  return JSON.stringify(value)
    .replace(/</g, '\\u003c')
    .replace(/\u2028/g, '\\u2028')
    .replace(/\u2029/g, '\\u2029')
}

function regexLiteral(pattern: RegExp): string {
  return `new RegExp(${scriptLiteral(pattern.source)},${scriptLiteral(pattern.flags)})`
}

export const SKIN_PREPAINT_SCRIPT = `(function(){
var d=document.documentElement,id=d.getAttribute(${scriptLiteral(SKIN_DATA_ATTRIBUTE)}),c=null;
if(!id||!${regexLiteral(SKIN_ID_PATTERN)}.test(id)){return;}
try{c=JSON.parse(window.localStorage.getItem(${scriptLiteral(SKIN_CSS_CACHE_KEY)})||"null");}catch(e){return;}
if(c===null||typeof c!=="object"||c.id!==id||typeof c.css!=="string"){return;}
if(!${regexLiteral(SKIN_CSS_GRAMMAR)}.test(c.css)){return;}
var own=c.css.split(${scriptLiteral(`${SKIN_DATA_ATTRIBUTE}="`)}).slice(1).every(function(x){return x.indexOf(id+'"')===0;});
if(!own){return;}
var s=document.getElementById(${scriptLiteral(SKIN_STYLE_ELEMENT_ID)});
if(!s){s=document.createElement("style");s.id=${scriptLiteral(SKIN_STYLE_ELEMENT_ID)};(document.head||d).appendChild(s);}
s.textContent=c.css;
})();`
