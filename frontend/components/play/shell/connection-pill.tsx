import type { HostHealth } from '@automatos/theme-sdk'

import styles from './connection-pill.module.css'

/**
 * PRD-260 T-004 — the connection pill at the foot of Play's sidebar (Ember mock
 * 01), ported from Automatos Play's components/shell/ConnectionPill.tsx. Fed by
 * the SDK's useHealth (GET /health, PRD-261 A-14f): Auto's light for connected,
 * the error colour for down, always with words.
 */

/** What the backend's health says: checking, connected or down. */
type BackendConnection = HostHealth['connection']

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
