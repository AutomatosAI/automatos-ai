'use client'

import type { ReactNode } from 'react'

interface StatusHeadProps {
  pip: string
  label: string
  count?: ReactNode
  right?: ReactNode
}

export function StatusHead({ pip, label, count, right }: StatusHeadProps) {
  return (
    <div className="status-head">
      <span className="pip" style={{ background: pip }} />
      <span className="l">{label}</span>
      {count != null && <span className="ct">· {count}</span>}
      {right && <span className="right">{right}</span>}
    </div>
  )
}
