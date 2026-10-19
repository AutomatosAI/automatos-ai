/**
 * PRD-261 A-01, A-10a — writes the ES-module shims a theme imports React, ReactDOM, the jsx runtime and
 * the theme SDK through: one file per entry of lib/theme-host/shared-modules.json, under its
 * publicDir (public/theme-sdk/v1), which the import map (lib/theme-host/import-map.ts) maps the
 * bare specifiers to.
 *
 * Each shim re-exports the host's own instance from the frozen global that
 * lib/theme-host/host-modules.ts installs, so a theme renders with the React the App Router
 * renders with. React's export names are read statically (`exports.<name> =`) from the CommonJS
 * builds Next vendors under next/dist/compiled, production and development together, so the lists
 * come from the install the build uses and nothing is executed. react/jsx-dev-runtime is not
 * shared: themes are built in production mode. The SDK's names (PRD-261 A-10a) are the union of
 * its surface files (lib/theme-sdk/surface/*.json), the same lists its surface test holds the
 * SDK module to; a name in two areas is refused, as `export *` would drop it.
 *
 * The shims are built, never committed (frontend/.gitignore): package.json's prebuild and predev
 * run this before every `next build` and `next dev`, so every build, and every image, serves
 * /theme-sdk/v1/*.
 *
 * Usage (from `frontend/`): node scripts/theme-sdk/generate-shims.js
 *
 * CommonJS on purpose, like csp-sources.js: node runs it directly, and its test requires it.
 * Every value written into a shim is checked against a strict pattern first and written between
 * plain double quotes.
 */
'use strict'

const fs = require('fs')
const path = require('path')
const sharedModules = require('../../lib/theme-host/shared-modules.json')

const EXPORT_ASSIGNMENT = /\bexports\.([A-Za-z_$][\w$]*)\s*=(?!=)/g
const SKIPPED_NAMES = new Set(['default', '__esModule'])
const SPECIFIER_PATTERN = /^[@a-z0-9/_-]+$/
const IDENTIFIER_PATTERN = /^[A-Za-z_$][\w$]*$/
const SHIM_FILE_PATTERN = /^[a-z0-9-]+\.js$/
const SURFACE_FILE_PATTERN = /^[a-z0-9-]+\.json$/
const FRONTEND_ROOT = path.resolve(__dirname, '..', '..')

/** The sorted, de-duplicated names a CommonJS build assigns with `exports.<name> =`. */
function exportNamesFrom(sourceText) {
  const names = Array.from(String(sourceText).matchAll(EXPORT_ASSIGNMENT), (match) => match[1])
  return Array.from(new Set(names.filter((name) => !SKIPPED_NAMES.has(name)))).sort()
}

function namesFromBuilds(files, frontendRoot) {
  const names = files.flatMap((file) => {
    const resolved = require.resolve(file, { paths: [frontendRoot] })
    const found = exportNamesFrom(fs.readFileSync(resolved, 'utf8'))
    if (found.length === 0) throw new Error(`${file} assigns no exports.<name>; cannot list its exports.`)
    return found
  })
  return Array.from(new Set(names)).sort()
}

function assertMatches(value, pattern, what) {
  if (typeof value !== 'string' || !pattern.test(value)) {
    throw new Error(`${what} ${String(value)} does not match ${pattern}.`)
  }
}

function readSurfaceNames(dir, file) {
  const surface = JSON.parse(fs.readFileSync(path.join(dir, file), 'utf8'))
  if (!surface || !Array.isArray(surface.names) || surface.names.length === 0) {
    throw new Error(`${file} lists no names.`)
  }
  surface.names.forEach((name) => assertMatches(name, IDENTIFIER_PATTERN, `export name in ${file}`))
  return surface.names
}

/** The sorted union of the names in every surface file under `surfaceDir`; a name in two files throws. */
function namesFromSurface(surfaceDir, frontendRoot) {
  const dir = path.resolve(frontendRoot, surfaceDir)
  const files = fs.readdirSync(dir).filter((file) => SURFACE_FILE_PATTERN.test(file)).sort()
  if (files.length === 0) throw new Error(`${surfaceDir} holds no surface files.`)
  const areaOf = new Map()
  for (const file of files) {
    for (const name of readSurfaceNames(dir, file)) {
      if (areaOf.has(name)) throw new Error(`${name} is in both ${areaOf.get(name)} and ${file}.`)
      areaOf.set(name, file)
    }
  }
  return Array.from(areaOf.keys()).sort()
}

function namesOf(entry, frontendRoot) {
  if (entry.from) return namesFromBuilds(entry.from, frontendRoot)
  if (entry.surface) return namesFromSurface(entry.surface, frontendRoot)
  throw new Error(`${entry.specifier} names neither its builds (from) nor its surface.`)
}

/** Each shared module with the names its shim exports: the union of its builds, or of its surface files. */
function sharedModuleExports(frontendRoot = FRONTEND_ROOT) {
  return sharedModules.modules.map((entry) => ({
    specifier: entry.specifier,
    file: entry.file,
    names: namesOf(entry, frontendRoot),
  }))
}

/** The shim's text for one shared module. Throws before writing anything unchecked. */
function renderShim(entry, names, hostGlobal) {
  assertMatches(entry.specifier, SPECIFIER_PATTERN, 'specifier')
  assertMatches(hostGlobal, IDENTIFIER_PATTERN, 'host global')
  names.forEach((name) => assertMatches(name, IDENTIFIER_PATTERN, 'export name'))
  const specifier = entry.specifier
  return [
    '// Generated by frontend/scripts/theme-sdk/generate-shims.js from the React that Next bundles',
    '// (next/dist/compiled). Do not edit. PRD-261 A-01.',
    `const host = globalThis.${hostGlobal};`,
    `const shared = host && host.modules ? host.modules["${specifier}"] : undefined;`,
    'if (!shared) {',
    `  throw new Error('Automatos theme host: "${specifier}" is not shared on this page.');`,
    '}',
    'export default shared.default ?? shared;',
    'export const {',
    ...names.map((name) => `  ${name},`),
    '} = shared;',
    '',
  ].join('\n')
}

function main() {
  try {
    const shims = sharedModuleExports(FRONTEND_ROOT).map((entry) => {
      assertMatches(entry.file, SHIM_FILE_PATTERN, 'shim file')
      return { file: entry.file, count: entry.names.length, text: renderShim(entry, entry.names, sharedModules.hostGlobal) }
    })
    const outDir = path.join(FRONTEND_ROOT, sharedModules.publicDir)
    fs.mkdirSync(outDir, { recursive: true })
    for (const shim of shims) {
      fs.writeFileSync(path.join(outDir, shim.file), shim.text)
      console.log(`wrote ${path.join(sharedModules.publicDir, shim.file)} (${shim.count} names)`)
    }
  } catch (error) {
    console.error(`generate-shims: ${error instanceof Error ? error.message : String(error)}`)
    process.exit(1)
  }
}

module.exports = { exportNamesFrom, namesFromSurface, sharedModuleExports, renderShim, main }

if (require.main === module) main()
