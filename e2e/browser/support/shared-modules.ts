import fs from 'fs'
import path from 'path'

// PRD-261 A-01: the host's shared-module table, read from the frontend (the one source for the
// global's name, the shims' public path and each shared specifier), so the specs check the page
// against the same table the host was built from.

export interface SharedModuleEntry {
  readonly specifier: string
  readonly file: string
}

export interface SharedModulesTable {
  readonly sdkVersion: string
  readonly hostGlobal: string
  readonly publicPath: string
  readonly modules: readonly SharedModuleEntry[]
}

const TABLE_PATH = path.resolve(__dirname, '..', '..', '..', 'frontend', 'lib', 'theme-host', 'shared-modules.json')

export function readSharedModules(): SharedModulesTable {
  return JSON.parse(fs.readFileSync(TABLE_PATH, 'utf8')) as SharedModulesTable
}

/** The import map the root layout must render for this table. */
export function expectedImportMap(table: SharedModulesTable): { imports: Record<string, string> } {
  return { imports: Object.fromEntries(table.modules.map((entry) => [entry.specifier, `${table.publicPath}${entry.file}`])) }
}
