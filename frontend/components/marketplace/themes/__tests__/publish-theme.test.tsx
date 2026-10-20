/**
 * PRD-261 C-01b — the web "Publish theme" form.
 *
 * The refusals are the wire shapes of `api/marketplace_themes_submit.py`: the package checks' 400
 * (and 413) `{ detail: { message, errors: [{ code, path, message }] } }` (`_read_upload`), the body-size
 * middleware's bare 413 (`main.py`), and a plain `detail` sentence for 403/409/429. `apiClient` attaches
 * the parsed body to the error it throws as `body` (lib/api-client.ts `httpErrorOf`).
 *
 * `@clerk/nextjs` is the same double as A-07d's tests: it does what Clerk's
 * `createReverificationHandler` does (call the fetcher; on Clerk's own `isReverificationHint`,
 * open the reverification and call the fetcher ONE more time), so a step-up 403 from submit is
 * proved to reach Clerk from the form.
 */
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const api = vi.hoisted(() => ({ submitTheme: vi.fn() }))
const clerk = vi.hoisted(() => ({ openReverification: vi.fn() }))

vi.mock('@/lib/api-client', () => ({ apiClient: api }))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspace: () => ({ workspace: { id: 'ws-1', name: 'Acme', role: 'member' } }),
}))
vi.mock('@/contexts/ui-style-context', () => ({ useUiStyleOptional: () => ({ setStyle: vi.fn() }) }))
vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }))
vi.mock('@clerk/nextjs', async () => {
  const { isReverificationHint } = await import('@clerk/shared/authorization-errors')
  type Fetcher = (...args: unknown[]) => Promise<unknown>
  return {
    useReverification: (fetcher: Fetcher) => async (...args: unknown[]) => {
      let result = await fetcher(...args)
      if (isReverificationHint(result)) {
        await clerk.openReverification()
        result = await fetcher(...args)
      }
      return result
    },
  }
})

import { STEP_UP_CANCELLED_MESSAGE } from '@/lib/theme-host/api'
import { stepUpError } from '@/components/settings/themes/__tests__/theme-fixtures'

import {
  EMPTY_PACKAGE,
  NOT_A_THEME_PACKAGE,
  PublishThemeForm,
  SUBMIT_FOR_REVIEW,
  packageFileProblem,
} from '../publish-theme'
import {
  PACKAGE_TOO_LARGE_MESSAGE,
  groupPackageErrors,
  publishErrorGroupOf,
  submissionRefusalOf,
} from '../publish-errors'
import { BLOCKED_NOTE, DUPLICATE_NOTE, PENDING_REVIEW, SUBMISSION_VERDICTS } from '../publish-result'

const PACKAGE_ID = '44444444-4444-4444-8444-444444444444'
// api/marketplace_themes_submit.py SUBMIT_REFUSED: the headline of a package-check refusal.
const SUBMIT_REFUSED = "This theme can't be submitted."

// sdk_uses.py UNKNOWN_NAMES and review_stage.py STALE_CLONES, as the server words them.
const UNKNOWN_NAMES = "The theme imports names this host's SDK doesn't export, so its code can't load: useSecret."
const STALE_CLONES =
  '1 cloned page(s) copied a core file this host has changed since, so the clone may miss fixes and features the core page has now.'
const USES_PASS = { level: 'pass', status: 'checked', problems: [], unknownNames: [], usedAreas: ['chat'], wideImport: false }
const USES_ERROR = { ...USES_PASS, level: 'error', problems: [UNKNOWN_NAMES], unknownNames: ['useSecret'] }

/**
 * review_stage.member_view of a run_review_stage report (schema v2): the reader checks keep
 * scan.py's `status` beside their `level`, `uses` says only `checked`, `clonedFrom` has no
 * status, and the LLM check and the level counts stay with the reviewer.
 */
function memberReport(verdict: string, uses: Record<string, unknown>) {
  return {
    schema: 'automatos-theme-scan/v2',
    stage: 'C-02a',
    verdict,
    checks: {
      sizes: { status: 'pass', level: 'pass', packageBytes: 20480, codeBytes: 10240, entryBytes: 2048 },
      cssScope: { status: 'pass', level: 'pass', path: 'dist/theme.css' },
      looks: { status: 'pass', level: 'pass', presets: ['default'] },
      contrast: { status: 'not_run', level: 'not_measured', reason: 'Not measured: no colour pairs.', presets: [] },
      uses,
      clonedFrom: { level: 'warning', clones: ['/chat'], stale: ['/chat'], reason: STALE_CLONES },
    },
  }
}

function submission(overrides: { scanVerdict?: string | null; duplicate?: boolean } = {}) {
  const scanVerdict = overrides.scanVerdict === undefined ? 'review_required' : overrides.scanVerdict
  return {
    package: {
      id: PACKAGE_ID, themeId: 'acme/hello', version: '1.0.0', name: 'Hello theme', digest: 'b'.repeat(64),
      ownerType: 'marketplace', status: 'pending', reviewed: false, scanVerdict,
      scanReport: memberReport(scanVerdict ?? 'not_run', scanVerdict === 'blocked' ? USES_ERROR : USES_PASS),
    },
    duplicate: overrides.duplicate ?? false,
  }
}

/** The error apiClient throws for a non-2xx answer: the parsed body rides it (plan §2.2). */
function httpError(status: number, body: unknown): Error {
  const detail = (body as { detail?: unknown }).detail
  return Object.assign(new Error(typeof detail === 'string' ? detail : JSON.stringify(detail)), { status, body })
}

function packageChecksError(status: number, errors: Array<{ code: string; path: string; message: string }>): Error {
  return httpError(status, { detail: { message: SUBMIT_REFUSED, errors } })
}

const MANIFEST_ERROR = { code: 'BAD_ENGINES', path: 'engines.sdk', message: '"engines.sdk" must be an exact version or a caret range.' }
const CSS_ERROR = { code: 'CSS_UNSCOPED', path: 'dist/theme.css', message: 'Every rule must start with html.hello.' }
const SIZE_ERROR = { code: 'ENTRY_TOO_LARGE', path: 'dist/entry.js', message: 'The entry module is over 300 KB.' }
const ZIP_ERROR = { code: 'NOT_A_ZIP', path: '', message: 'The package is not a zip archive.' }
const LOOK_ERROR = { code: 'NO_PRESET', path: 'looks/presets', message: 'A theme ships at least one preset.' }

function renderForm() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <PublishThemeForm />
    </QueryClientProvider>,
  )
}

function choose(name = 'hello.autotheme', content = 'zip'): File {
  const file = new File([content], name, { type: 'application/zip' })
  fireEvent.change(screen.getByTestId('publish-theme-input'), { target: { files: [file] } })
  return file
}

function submit() {
  fireEvent.click(screen.getByRole('button', { name: SUBMIT_FOR_REVIEW }))
}

beforeEach(() => {
  api.submitTheme.mockReset()
  clerk.openReverification.mockReset()
  clerk.openReverification.mockResolvedValue(undefined)
})

describe('Publish theme form', () => {
  it('submits the chosen package through apiClient.submitTheme and shows it pending review with its verdict', async () => {
    api.submitTheme.mockResolvedValue(submission())
    renderForm()
    const file = choose()
    submit()

    const result = await screen.findByTestId('publish-result')
    expect(api.submitTheme).toHaveBeenCalledTimes(1)
    expect(api.submitTheme).toHaveBeenCalledWith(file)
    expect(within(result).getByText('Hello theme')).toBeInTheDocument()
    expect(within(result).getByText(PENDING_REVIEW)).toBeInTheDocument()
    expect(within(result).getByText(/Verdict:/).textContent).toContain(SUBMISSION_VERDICTS.review_required)
    // The v2 report is read by its levels: the stale clone is a Warning with its reason, never "unknown".
    const clone = within(result).getByTestId('theme-check-clonedFrom')
    expect(within(clone).getByText('Warning')).toHaveAttribute('data-level', 'warning')
    expect(within(clone).getByText(STALE_CLONES)).toBeInTheDocument()
    expect(within(within(result).getByTestId('theme-check-uses')).getByText('Pass')).toBeInTheDocument()
    expect(within(within(result).getByTestId('theme-check-contrast')).getByText('Not measured')).toBeInTheDocument()
    expect(within(result).queryByText('unknown')).not.toBeInTheDocument()
    expect(within(result).queryByText('checked')).not.toBeInTheDocument()
    expect(within(result).queryByTestId('publish-blocked')).not.toBeInTheDocument()
    expect(within(result).queryByText(new RegExp(DUPLICATE_NOTE))).not.toBeInTheDocument()
  })

  it('says when these exact bytes were submitted before, and when the stage blocked the version', async () => {
    api.submitTheme.mockResolvedValue(submission({ scanVerdict: 'blocked', duplicate: true }))
    renderForm()
    choose()
    submit()

    const result = await screen.findByTestId('publish-result')
    expect(within(result).getByText(new RegExp(DUPLICATE_NOTE))).toBeInTheDocument()
    expect(within(result).getByTestId('publish-blocked')).toHaveTextContent(BLOCKED_NOTE)
    // Which check blocked it, and why: the `uses` error with the server's problem sentence.
    const uses = within(result).getByTestId('theme-check-uses')
    expect(within(uses).getByText('Error')).toHaveAttribute('data-level', 'error')
    expect(within(uses).getByText(UNKNOWN_NAMES)).toBeInTheDocument()
    expect(within(result).getByText(/Verdict:/).textContent).toContain(SUBMISSION_VERDICTS.blocked)
  })

  it('lists every package-check error of a 400 under its part of the package, with path and code', async () => {
    api.submitTheme.mockRejectedValue(packageChecksError(400, [MANIFEST_ERROR, CSS_ERROR, SIZE_ERROR, ZIP_ERROR, LOOK_ERROR]))
    renderForm()
    choose()
    submit()

    const errors = await screen.findByTestId('publish-errors')
    expect(errors).toHaveAttribute('role', 'alert')
    expect(within(errors).getByText(SUBMIT_REFUSED)).toBeInTheDocument()
    const expected: Array<[string, typeof MANIFEST_ERROR]> = [
      ['manifest', MANIFEST_ERROR], ['styleSheet', CSS_ERROR], ['sizes', SIZE_ERROR], ['archive', ZIP_ERROR], ['looks', LOOK_ERROR],
    ]
    for (const [group, error] of expected) {
      const section = within(errors).getByTestId(`publish-errors-${group}`)
      expect(within(section).getByText(error.message)).toBeInTheDocument()
      expect(within(section).getByText(error.code)).toBeInTheDocument()
      if (error.path) expect(section.textContent).toContain(error.path)
    }
    expect(screen.queryByTestId('publish-result')).not.toBeInTheDocument()
  })

  it('lists the errors of the route’s own 413 (PACKAGE_TOO_LARGE) and words the middleware’s bare 413', async () => {
    const tooLarge = { code: 'PACKAGE_TOO_LARGE', path: '', message: 'The package is over 5 MB.' }
    api.submitTheme.mockRejectedValueOnce(packageChecksError(413, [tooLarge]))
    api.submitTheme.mockRejectedValueOnce(httpError(413, { detail: 'Payload too large' }))
    renderForm()
    choose()
    submit()
    const listed = await screen.findByTestId('publish-errors-sizes')
    expect(within(listed).getByText(tooLarge.message)).toBeInTheDocument()

    submit()
    await waitFor(() => expect(api.submitTheme).toHaveBeenCalledTimes(2))
    expect(await screen.findByText(PACKAGE_TOO_LARGE_MESSAGE)).toBeInTheDocument()
  })

  it('shows the server’s sentence for a refusal that is not a package check (409 version clash)', async () => {
    const clash = 'Version 1.0.0 of acme/hello was submitted already: raise the version and submit again.'
    api.submitTheme.mockRejectedValue(httpError(409, { detail: clash }))
    renderForm()
    choose()
    submit()

    expect(await screen.findByTestId('publish-refused')).toHaveTextContent(clash)
    expect(screen.queryByTestId('publish-errors')).not.toBeInTheDocument()
  })

  it('refuses a file that is not a .autotheme, or is empty, without calling the server', () => {
    renderForm()
    choose('hello.zip')
    expect(screen.getByTestId('publish-file-problem')).toHaveTextContent(NOT_A_THEME_PACKAGE)
    expect(screen.getByRole('button', { name: SUBMIT_FOR_REVIEW })).toBeDisabled()

    choose('hello.autotheme', '')
    expect(screen.getByTestId('publish-file-problem')).toHaveTextContent(EMPTY_PACKAGE)
    submit()
    expect(api.submitTheme).not.toHaveBeenCalled()
  })

  it('takes a step-up 403 to Clerk’s reverification and submits once more', async () => {
    api.submitTheme.mockRejectedValueOnce(stepUpError()).mockResolvedValueOnce(submission())
    renderForm()
    choose()
    submit()

    await screen.findByTestId('publish-result')
    expect(clerk.openReverification).toHaveBeenCalledTimes(1)
    expect(api.submitTheme).toHaveBeenCalledTimes(2)
  })

  it('says nothing was changed when the person closes the confirmation', async () => {
    clerk.openReverification.mockRejectedValue(Object.assign(new Error('closed'), { code: 'reverification_cancelled' }))
    api.submitTheme.mockRejectedValue(stepUpError())
    renderForm()
    choose()
    submit()

    expect(await screen.findByTestId('publish-refused')).toHaveTextContent(STEP_UP_CANCELLED_MESSAGE)
    expect(api.submitTheme).toHaveBeenCalledTimes(1)
  })
})

describe('the 400 codes', () => {
  it('groups each code by the part of the package that raised it', () => {
    expect(publishErrorGroupOf('BAD_ENGINES')).toBe('manifest')
    expect(publishErrorGroupOf('RESERVED_HTML_CLASS')).toBe('manifest')
    expect(publishErrorGroupOf('BAD_LOOK_ID')).toBe('manifest')
    expect(publishErrorGroupOf('BAD_MOBILE')).toBe('manifest')
    expect(publishErrorGroupOf('MOBILE_CANNOT_PAINT')).toBe('manifest')
    expect(publishErrorGroupOf('CSS_IMPORT')).toBe('styleSheet')
    expect(publishErrorGroupOf('LOOK_BAD_VALUE')).toBe('looks')
    expect(publishErrorGroupOf('BAD_KNOB')).toBe('looks')
    expect(publishErrorGroupOf('UNKNOWN_DEFAULT_LOOK')).toBe('looks')
    expect(publishErrorGroupOf('CODE_TOO_LARGE')).toBe('sizes')
    expect(publishErrorGroupOf('TOO_MANY_FILES')).toBe('sizes')
    expect(publishErrorGroupOf('SYMLINK')).toBe('archive')
    expect(publishErrorGroupOf('A_CODE_ADDED_LATER')).toBe('archive')
  })

  it('keeps every error, in the form’s heading order, leaving out empty headings', () => {
    const groups = groupPackageErrors([LOOK_ERROR, ZIP_ERROR, MANIFEST_ERROR])
    expect(groups.map((group) => group.id)).toEqual(['archive', 'manifest', 'looks'])
    expect(groups.flatMap((group) => group.errors)).toHaveLength(3)
  })

  it('reads only a 400/413 package-check body; anything else is the server’s sentence', () => {
    expect(submissionRefusalOf(packageChecksError(400, [ZIP_ERROR]))).toEqual({ message: SUBMIT_REFUSED, errors: [ZIP_ERROR] })
    expect(submissionRefusalOf(httpError(413, { detail: 'Payload too large' }))).toEqual({
      message: PACKAGE_TOO_LARGE_MESSAGE, errors: [],
    })
    expect(submissionRefusalOf(httpError(400, { detail: 'Invalid Content-Length header' }))).toBeNull()
    expect(submissionRefusalOf(httpError(409, { detail: { message: SUBMIT_REFUSED, errors: [ZIP_ERROR] } }))).toBeNull()
    expect(submissionRefusalOf(stepUpError())).toBeNull()
    expect(submissionRefusalOf(new Error('network'))).toBeNull()
  })

  it('checks the name and size it can see before calling', () => {
    expect(packageFileProblem(new File(['x'], 'Hello.AUTOTHEME'))).toBeNull()
    expect(packageFileProblem(new File(['x'], 'hello.autotheme.zip'))).toBe(NOT_A_THEME_PACKAGE)
    expect(packageFileProblem(new File([], 'hello.autotheme'))).toBe(EMPTY_PACKAGE)
  })
})
