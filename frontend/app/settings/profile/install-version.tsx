'use client'

/**
 * PRD-279 US-303 — the edition and the version the API reports, on the profile page, where the
 * web app names the edition. The version is `version` in GET /api/system/health's answer
 * (orchestrator/api/system.py, read through apiClient once per visit): the tag's version on a
 * release, the commit id on any other published build, `dev` on a build given none. Each edition
 * shows it the same way.
 *
 * Nothing is shown until the API has answered with a value of that form, so a failed read or an
 * odd answer never puts a made-up version on the page. The read is logged when it fails.
 */
import { useEffect, useState } from 'react'

import { apiClient } from '@/lib/api-client'
import { authEdition, type AuthEdition } from '@/lib/auth-edition'

export const EDITION_LABELS: Readonly<Record<AuthEdition, string>> = {
  local: 'Local edition',
  saas: 'Hosted edition',
  enterprise: 'Enterprise edition',
}

/** A release's version or a commit id: scripts/ci/image_matrix.py's BUILD_VERSION. */
const VERSION_SHAPE = /^[0-9A-Za-z][0-9A-Za-z.-]{0,127}$/

/** The version in an answer of GET /api/system/health, or null when it holds none of that form. */
export function reportedVersion(answer: unknown): string | null {
  if (typeof answer !== 'object' || answer === null) return null
  const version = (answer as { version?: unknown }).version
  return typeof version === 'string' && VERSION_SHAPE.test(version) ? version : null
}

/** The version the API reports, read once; null until it has answered with one. */
export function useInstallVersion(): string | null {
  const [version, setVersion] = useState<string | null>(null)
  useEffect(() => {
    let mounted = true
    const read = async () => {
      try {
        const answer = await apiClient.getSystemHealth()
        if (mounted) setVersion(reportedVersion(answer))
      } catch (error) {
        console.warn('[version] the API did not report its version:', error)
      }
    }
    void read()
    return () => {
      mounted = false
    }
  }, [])
  return version
}

/** "<Edition> · version <v>", once the API has reported its version. */
export function InstallVersion({ edition = authEdition }: { edition?: AuthEdition }) {
  const version = useInstallVersion()
  if (!version) return null
  return (
    <p className="mt-3 text-xs text-muted-foreground" data-testid="install-version">
      {EDITION_LABELS[edition]} · version <span className="font-mono">{version}</span>
    </p>
  )
}
