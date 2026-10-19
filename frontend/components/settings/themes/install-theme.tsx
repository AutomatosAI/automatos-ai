'use client'

/**
 * PRD-261 A-07d — Install theme: upload a `.autotheme` into this workspace (PRD §5).
 *
 * `POST /api/themes/import` (manage, who may install, the private-upload switch and setting,
 * step-up). The answer carries the package and its scan report, shown here with "Private · not
 * reviewed"; the install appears in the list below, where each member opts in and chooses it. A
 * refusal (package checks, a version clash, uploads off) shows the server's sentence. The button
 * is reused by the Marketplace's Install theme (W-C2, C-04b), not copied.
 */
import { useRef } from 'react'
import { Upload } from 'lucide-react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { useImportTheme } from '@/hooks/use-installed-themes'
import type { ThemeSettings, ThemeUpload } from '@/lib/api-client'

import { ScanReport } from './scan-report'
import { PRIVATE_NOT_REVIEWED } from './theme-labels'

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

/** Why this caller can't upload here, or null when the button works. */
export function uploadBlockedReason(settings: ThemeSettings | undefined): string | null {
  if (!settings) return null
  if (!settings.canInstall) {
    return `In this workspace only ${WHO_INSTALLS[settings.installRole] ?? WHO_INSTALLS.owner} can install themes.`
  }
  return settings.privateUploads.effective === 'off' ? UPLOADS_OFF : null
}

/** A note that does not block: under the platform's `admins` switch the server decides who uploads. */
function uploadNote(settings: ThemeSettings | undefined): string | null {
  return settings?.privateUploads.effective === 'admins' ? UPLOADS_PLATFORM_ADMINS : null
}

function UploadResult({ upload }: { upload: ThemeUpload }) {
  const pkg = upload.package
  return (
    <div className="space-y-3 rounded-xl border border-border/50 px-4 py-3" data-testid="theme-upload-result">
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-medium text-foreground">{pkg.name || pkg.themeId}</span>
        <span className="text-xs text-muted-foreground">{pkg.version}</span>
        {!pkg.reviewed && <Badge variant="secondary">{PRIVATE_NOT_REVIEWED}</Badge>}
      </div>
      <p className="text-xs text-muted-foreground">
        {STATUS_TEXT[upload.status]}
        {upload.duplicate ? ' These exact files were uploaded before.' : ''}
      </p>
      <ScanReport report={pkg.scanReport} verdict={pkg.scanVerdict} />
    </div>
  )
}

export function InstallTheme({ settings }: { settings: ThemeSettings | undefined }) {
  const input = useRef<HTMLInputElement>(null)
  const importTheme = useImportTheme()
  const blocked = uploadBlockedReason(settings)
  const note = blocked ?? uploadNote(settings)

  const onFile = (file: File | undefined) => {
    if (file) importTheme.mutate(file)
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
      {importTheme.data && <UploadResult upload={importTheme.data} />}
    </section>
  )
}
