'use client'

/**
 * PRD-261 C-01b — the web "Publish theme" form: submit a `.autotheme` to the Marketplace for review.
 *
 * `apiClient.submitTheme` → `POST /api/marketplace/themes/submit` (member, the Marketplace switch,
 * step-up; plan §2.2). The call goes through `useStepUp` (lib/theme-host/api.ts), so in SaaS Clerk
 * asks the person to confirm it's them and the call runs once more; the local edition calls
 * straight through. The server checks the package and runs the automated review stage before it
 * answers, so the button says so while it waits.
 *
 * Before calling, the form refuses only what it can see without reading the file: a name that is
 * not a `.autotheme` and an empty file. Every other rule (size caps, manifest, style sheet, looks,
 * publisher ownership, versions, the daily quota) is the server's: a 400/413 lists the package
 * checks' errors (publish-errors.tsx); any other refusal shows the server's sentence.
 */
import { useRef, useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { Send, Upload } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { THEME_PACKAGE_ACCEPT } from '@/components/settings/themes/install-theme'
import { apiClient } from '@/lib/api-client'
import type { ThemeSubmission } from '@/lib/api-client'
import { themeErrorMessage, useStepUp } from '@/lib/theme-host/api'

import { PublishErrors, submissionRefusalOf } from './publish-errors'
import { PublishResult } from './publish-result'

export const PUBLISH_THEME = 'Publish theme'
export const SUBMIT_FOR_REVIEW = 'Submit for review'
export const SUBMITTING = 'Checking and reviewing the package…'
export const CHOOSE_PACKAGE = 'Choose a .autotheme file'
export const NOT_A_THEME_PACKAGE = 'Choose a .autotheme file: the theme kit builds one from your theme.'
export const EMPTY_PACKAGE = 'This file is empty.'
export const PUBLISH_INTRO =
  'Submit your theme package for review. Once a person at Automatos approves it, every workspace can install it from the Marketplace.'

/** Why `file` can't be submitted, as far as its name and size show, or null. */
export function packageFileProblem(file: File): string | null {
  if (!file.name.toLowerCase().endsWith(THEME_PACKAGE_ACCEPT)) return NOT_A_THEME_PACKAGE
  return file.size === 0 ? EMPTY_PACKAGE : null
}

/** The submit call, behind step-up; refusals stay on the mutation for the form to show. */
export function usePublishTheme() {
  const submitTheme = useStepUp(apiClient.submitTheme.bind(apiClient))
  return useMutation<ThemeSubmission, unknown, File>({ mutationFn: (file) => submitTheme(file) })
}

function PublishRefusal({ error }: { error: unknown }) {
  const refusal = submissionRefusalOf(error)
  if (refusal) return <PublishErrors refusal={refusal} />
  return (
    <p role="alert" className="text-sm text-destructive" data-testid="publish-refused">
      {themeErrorMessage(error)}
    </p>
  )
}

export function PublishThemeForm() {
  const input = useRef<HTMLInputElement>(null)
  const [file, setFile] = useState<File | null>(null)
  const publish = usePublishTheme()
  const problem = file ? packageFileProblem(file) : null
  const pending = publish.isLoading

  const onFile = (chosen: File | undefined) => {
    setFile(chosen ?? null)
    publish.reset()
    if (input.current) input.current.value = ''
  }
  const onSubmit = () => {
    if (file && !problem) publish.mutate(file)
  }

  return (
    <section aria-label={PUBLISH_THEME} className="space-y-4" data-testid="publish-theme">
      <p className="text-sm text-muted-foreground">{PUBLISH_INTRO}</p>
      <div className="flex flex-wrap items-center gap-3">
        <Button variant="outline" onClick={() => input.current?.click()} disabled={pending}>
          <Upload className="mr-2 h-4 w-4" aria-hidden />
          {CHOOSE_PACKAGE}
        </Button>
        <input
          ref={input}
          type="file"
          accept={THEME_PACKAGE_ACCEPT}
          className="hidden"
          data-testid="publish-theme-input"
          onChange={(event) => onFile(event.target.files?.[0])}
        />
        {file && <span className="text-sm text-foreground" data-testid="publish-theme-file">{file.name}</span>}
      </div>
      {problem && (
        <p role="alert" className="text-sm text-destructive" data-testid="publish-file-problem">
          {problem}
        </p>
      )}
      <Button onClick={onSubmit} disabled={!file || problem !== null || pending}>
        <Send className="mr-2 h-4 w-4" aria-hidden />
        {pending ? SUBMITTING : SUBMIT_FOR_REVIEW}
      </Button>
      {publish.isError && <PublishRefusal error={publish.error} />}
      {publish.data && <PublishResult submission={publish.data} />}
    </section>
  )
}
