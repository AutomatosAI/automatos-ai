'use client'

/**
 * PRD-261 A-07d — Install theme: upload a `.autotheme` into this workspace (PRD §5).
 *
 * `POST /api/themes/import` (manage, who may install, the private-upload switch and setting,
 * step-up). The answer carries the package and its scan report, shown here with "Private · not
 * reviewed"; the install appears in the list below, where each member opts in and chooses it. A
 * refusal (package checks, a version clash, uploads off) shows the server's sentence. The button
 * is reused by the Marketplace's Install theme (W-C2, C-04b), not copied.
 *
 * C-05k: before choosing the package, the admin may add its detached signature file (`.sig.json`,
 * released beside the `.autotheme`); both go in one upload, the file as `signature` (C-05d). The
 * server checks it against the package's own files before it stores anything: a refusal
 * (`SIGNATURE_MISMATCH` for another package's file, `SIGNATURE_BAD`, 413 over 4 KB) is listed here
 * in the server's own sentences, and nothing is installed. An upload that carried a signature and
 * was taken is signed: it shows "Signed by Automatos" in place of "Private · not reviewed" (QB-24).
 */
import { useRef, useState } from 'react'
import { ShieldCheck, Upload } from 'lucide-react'

import { submissionRefusalOf } from '@/components/marketplace/themes/publish-errors'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { useImportTheme } from '@/hooks/use-installed-themes'
import type { ThemeSettings, ThemeUpload } from '@/lib/api-client'

import { ScanReport } from './scan-report'
import {
  ADD_SIGNATURE_FILE,
  PRIVATE_NOT_REVIEWED,
  REMOVE_SIGNATURE_FILE,
  SIGNATURE_FILE_ACCEPT,
  SIGNATURE_FILE_CHOSEN,
  SIGNATURE_FILE_HINT,
  SIGNED_BY_AUTOMATOS,
} from './theme-labels'

export const INSTALL_THEME = 'Install theme'
export const THEME_PACKAGE_ACCEPT = '.autotheme'
export const UPLOADS_OFF = 'Private theme uploads are off in this workspace.'
export const UPLOADS_PLATFORM_ADMINS = 'On this platform only its administrators upload private themes.'

const WHO_INSTALLS: Readonly<Record<string, string>> = {
  admins: 'a workspace owner or admin',
  owner: 'the workspace owner',
}

const STATUS_TEXT: Readonly<Record<ThemeUpload['status'], string>> = {
  installed: 'Installed in this workspace.',
  update_available: 'Stored as an update to the installed version: review it in the list below.',
  stored: 'Stored. It is not installed.',
}

/**
 * Why this caller can't install any theme here (the workspace's "who installs", the server's
 * `require_install_role`), or null. Also the Marketplace's "Add to my workspace" gate (C-04b).
 */
export function installRoleRefusal(settings: ThemeSettings | undefined): string | null {
  if (!settings || settings.canInstall) return null
  return `In this workspace only ${WHO_INSTALLS[settings.installRole] ?? WHO_INSTALLS.owner} can install themes.`
}

/** Why this caller can't upload here, or null when the button works. */
export function uploadBlockedReason(settings: ThemeSettings | undefined): string | null {
  if (!settings) return null
  return installRoleRefusal(settings) ?? (settings.privateUploads.effective === 'off' ? UPLOADS_OFF : null)
}

/** A note that does not block: under the platform's `admins` switch the server decides who uploads. */
function uploadNote(settings: ThemeSettings | undefined): string | null {
  return settings?.privateUploads.effective === 'admins' ? UPLOADS_PLATFORM_ADMINS : null
}

function UploadResult({ upload, signed }: { upload: ThemeUpload; signed: boolean }) {
  const pkg = upload.package
  return (
    <div className="space-y-3 rounded-xl border border-border/50 px-4 py-3" data-testid="theme-upload-result">
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-medium text-foreground">{pkg.name || pkg.themeId}</span>
        <span className="text-xs text-muted-foreground">{pkg.version}</span>
        {signed && <Badge variant="secondary">{SIGNED_BY_AUTOMATOS}</Badge>}
        {!signed && !pkg.reviewed && <Badge variant="secondary">{PRIVATE_NOT_REVIEWED}</Badge>}
      </div>
      <p className="text-xs text-muted-foreground">
        {STATUS_TEXT[upload.status]}
        {upload.duplicate ? ' These exact files were uploaded before.' : ''}
      </p>
      <ScanReport report={pkg.scanReport} verdict={pkg.scanVerdict} />
    </div>
  )
}

/** The checks' refusal of an upload (400 or 413 with `{message, errors}`), each in the server's words. */
function UploadRefusal({ error }: { error: unknown }) {
  const refusal = submissionRefusalOf(error)
  if (refusal === null) return null
  return (
    <div role="alert" className="space-y-1 rounded-xl border border-destructive/40 px-4 py-3" data-testid="theme-upload-refusal">
      <p className="text-sm font-medium text-destructive">{refusal.message}</p>
      <ul className="space-y-1">
        {refusal.errors.map((item, index) => (
          <li key={`${item.code}-${index}`} className="text-sm">
            {item.message} <code className="text-xs text-muted-foreground">{item.code}</code>
          </li>
        ))}
      </ul>
    </div>
  )
}

function SignatureFile({ file, disabled, onChange }: { file: File | null; disabled: boolean; onChange: (next: File | null) => void }) {
  const input = useRef<HTMLInputElement>(null)
  return (
    <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
      {file ? (
        <>
          <span data-testid="install-theme-signature-name">
            {SIGNATURE_FILE_CHOSEN} <span className="font-mono text-foreground">{file.name}</span>
          </span>
          <Button type="button" size="sm" variant="ghost" disabled={disabled} onClick={() => onChange(null)}>
            {REMOVE_SIGNATURE_FILE}
          </Button>
        </>
      ) : (
        <Button type="button" size="sm" variant="outline" disabled={disabled} onClick={() => input.current?.click()}>
          <ShieldCheck className="mr-2 h-4 w-4" aria-hidden />
          {ADD_SIGNATURE_FILE}
        </Button>
      )}
      <span>{SIGNATURE_FILE_HINT}</span>
      <input
        ref={input}
        type="file"
        accept={SIGNATURE_FILE_ACCEPT}
        className="hidden"
        data-testid="install-theme-signature-input"
        onChange={(event) => {
          onChange(event.target.files?.[0] ?? null)
          event.target.value = ''
        }}
      />
    </div>
  )
}

export function InstallTheme({ settings }: { settings: ThemeSettings | undefined }) {
  const input = useRef<HTMLInputElement>(null)
  const importTheme = useImportTheme()
  const [signature, setSignature] = useState<File | null>(null)
  const blocked = uploadBlockedReason(settings)
  const note = blocked ?? uploadNote(settings)

  const onFile = (file: File | undefined) => {
    if (file) importTheme.mutate({ file, signature }, { onSuccess: () => setSignature(null) })
    if (input.current) input.current.value = ''
  }

  return (
    <section aria-label={INSTALL_THEME} className="space-y-3">
      <div className="flex flex-wrap items-center gap-3">
        <Button onClick={() => input.current?.click()} disabled={blocked !== null || importTheme.isLoading}>
          <Upload className="mr-2 h-4 w-4" aria-hidden />
          {importTheme.isLoading ? 'Checking the package…' : INSTALL_THEME}
        </Button>
        <input
          ref={input}
          type="file"
          accept={THEME_PACKAGE_ACCEPT}
          className="hidden"
          data-testid="install-theme-input"
          onChange={(event) => onFile(event.target.files?.[0])}
        />
        {note && <p className="text-xs text-muted-foreground">{note}</p>}
      </div>
      {blocked === null && <SignatureFile file={signature} disabled={importTheme.isLoading} onChange={setSignature} />}
      {importTheme.error ? <UploadRefusal error={importTheme.error} /> : null}
      {importTheme.data && (
        <UploadResult upload={importTheme.data} signed={Boolean(importTheme.variables?.signature)} />
      )}
    </section>
  )
}
