import type { BackendConnection } from '@/hooks/use-backend-health'

import styles from './connection-pill.module.css'

/**
 * PRD-260 T-004 — the connection pill at the foot of Play's sidebar (Ember mock
 * 01), ported from Automatos Play's components/shell/ConnectionPill.tsx. Fed by
 * useBackendHealth (GET /health): Auto's light for connected, the error colour
 * for down, always with words.
 */

export function connectionLabel(connection: BackendConnection, host?: string): string {
  switch (connection) {
    case 'connected':
      return host ? `Connected · ${host}` : 'Connected'
    case 'down':
      return 'Not running'
    case 'checking':
      return 'Checking…'
  }
}

export interface ConnectionPillProps {
  readonly connection: BackendConnection
  readonly host?: string
}

export function ConnectionPill({ connection, host }: ConnectionPillProps) {
  return (
    <div className={styles.connection} role="status">
      <span className={styles.dot} data-status={connection} aria-hidden="true" />
      {connectionLabel(connection, host)}
    </div>
  )
}
