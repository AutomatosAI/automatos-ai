import { execFile, type ExecFileException } from 'child_process'
import fs from 'fs'
import path from 'path'
import { expect, type APIRequestContext } from '@playwright/test'

// PRD-261 B-03h: a Design Kit theme installed in the local edition by the kit's own CLI, the way a
// theme author installs one (PRD §8 item 7):
//
//   automatos-theme install <file.autotheme> --to http://localhost:3000 --choose
//
// `--to` is the web app's URL: the CLI reads its /automatos-host.json (B-03j) to find the API, then
// imports the package, opts in to it and, with --choose, makes it the member's theme (the kit's
// packages/theme-cli/src/install.ts), with no token, as the local edition needs none. Nothing is
// intercepted. The lane's `package` job packs the run's theme (a dispatch's kit_theme, else Play) and
// the browser job downloads it into E2E_KIT_PACKAGE_DIR; the browser job also checks the kit out at
// the same ref and installs it from its lockfile with no install scripts, so E2E_KIT_CLI is the kit's
// own command (theme-cli's package.json `bin`, TypeScript that this Node runs as it is).

const BACKEND_URL = process.env.E2E_BACKEND_URL ?? 'http://localhost:8000'
/** The folder the lane downloaded the run's one package into; empty when the lane named none. */
const KIT_PACKAGE_DIR = process.env.E2E_KIT_PACKAGE_DIR ?? ''
/** The kit's `automatos-theme` command; empty when the lane named none. */
const KIT_CLI = process.env.E2E_KIT_CLI ?? ''
/** frontend/app/automatos-host.json/route.ts: the web app's answer to where its API is. */
const HOST_FILE_ROUTE = '/automatos-host.json'
/** The whole install, at most: the import waits on the server's review stage, the slowest step. */
export const CLI_TIMEOUT_MS = 120_000
const PACKAGE_SUFFIX = '.autotheme'
const INSTALLED_STYLE_PREFIX = 'theme:install:'
const EXIT_OK = 0
const OK = 200
/** The CLI's credentials (the kit's submit-target.ts), left blank: the local edition takes none. */
const NO_CREDENTIALS = { AUTOMATOS_API_TOKEN: '', AUTOMATOS_WORKSPACE_ID: '' }

/** What the host reads of a theme.json; the server's active answer carries it as the package has it. */
export interface ThemeManifest {
  readonly htmlClass?: unknown
  readonly pages?: unknown
  readonly routes?: unknown
}

interface ActiveAnswer {
  readonly choice?: unknown
  readonly theme?: {
    readonly installId?: unknown
    readonly themeId?: unknown
    readonly manifest?: ThemeManifest
    readonly optIn?: { readonly required?: unknown } | null
  } | null
}

/** The run's theme, as the CLI installed and chose it. */
export interface CliInstall {
  readonly installId: string
  readonly themeId: string
  readonly manifest: ThemeManifest
  /** `theme:install:<installId>`, the style cookie the choice writes. */
  readonly style: string
}

interface CliRun {
  /** The exit code; null when the CLI was killed (its time ran out) or never started. */
  readonly code: number | null
  readonly stdout: string
  readonly stderr: string
}

/** The run's one package; none, or more than one, is a lane failure, never a skip. */
function kitThemePackage(): string {
  expect(KIT_PACKAGE_DIR, 'E2E_KIT_PACKAGE_DIR is set by the lane step that runs the @installed spec').not.toBe('')
  const found = fs.existsSync(KIT_PACKAGE_DIR) ? fs.readdirSync(KIT_PACKAGE_DIR) : []
  const packages = found.filter((name) => name.endsWith(PACKAGE_SUFFIX))
  expect(packages, `the package job's one package was downloaded into ${KIT_PACKAGE_DIR}`).toHaveLength(1)
  return path.join(KIT_PACKAGE_DIR, packages[0])
}

function exitCode(error: ExecFileException | null): number | null {
  if (error === null) return EXIT_OK
  return typeof error.code === 'number' ? error.code : null
}

/** Runs the kit's CLI with this Node and resolves with how it ended; it never rejects. */
function runCli(args: readonly string[]): Promise<CliRun> {
  expect(KIT_CLI, 'E2E_KIT_CLI is set by the lane step that runs the @installed spec').not.toBe('')
  expect(fs.existsSync(KIT_CLI), `the kit's CLI was checked out and installed at ${KIT_CLI}`).toBe(true)
  const options = { env: { ...process.env, ...NO_CREDENTIALS }, timeout: CLI_TIMEOUT_MS, encoding: 'utf8' as const }
  return new Promise<CliRun>((resolve) => {
    execFile(process.execPath, [KIT_CLI, ...args], options, (error, stdout, stderr) => {
      resolve({ code: exitCode(error), stdout, stderr })
    })
  })
}

async function activeTheme(request: APIRequestContext): Promise<ActiveAnswer> {
  const response = await request.get(`${BACKEND_URL}/api/themes/active`)
  expect(response.status(), `GET /api/themes/active answered ${response.status()}: ${await response.text()}`).toBe(OK)
  return (await response.json()) as ActiveAnswer
}

/**
 * Installs the run's package with `automatos-theme install <file> --to <appUrl> --choose` and reads
 * back what the server now gives this member. The CLI exits 0 only once the import, the opt-in and
 * the choice all succeeded, and names the host file when that is where it found the API.
 */
export async function installWithKitCli(request: APIRequestContext, appUrl: string): Promise<CliInstall> {
  const run = await runCli(['install', kitThemePackage(), '--to', appUrl, '--choose'])
  const printed = `automatos-theme install exited ${run.code}:\n${run.stdout}${run.stderr}`
  expect(run.code, printed).toBe(EXIT_OK)
  expect(run.stdout, `the CLI found the API through the web app's host file. ${printed}`).toContain(`${appUrl}${HOST_FILE_ROUTE}`)
  const active = await activeTheme(request)
  const installId = String(active.theme?.installId ?? '')
  expect(installId, 'the server answers an installed theme').not.toBe('')
  expect(active.choice, "--choose stored the install as the member's choice").toBe(installId)
  expect(active.theme?.optIn?.required, 'the CLI opted the member in to the private package').toBe(false)
  return {
    installId,
    themeId: String(active.theme?.themeId ?? ''),
    manifest: active.theme?.manifest ?? {},
    style: `${INSTALLED_STYLE_PREFIX}${installId}`,
  }
}
