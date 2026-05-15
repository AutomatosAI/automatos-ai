'use client'

import { Download, BookMarked, Rocket } from 'lucide-react'
import type { RecommendedItem } from '@/hooks/use-assignments-api'

interface MktCardProps {
  item: RecommendedItem
  onClick: (item: RecommendedItem) => void
}

export function MktCard({ item, onClick }: MktCardProps) {
  const Icon = item.type === 'mission' ? Rocket : BookMarked
  return (
    <button type="button" className="mkt-card" onClick={() => onClick(item)}>
      <div className="top">
        <span className="ic">
          <Icon style={{ width: 14, height: 14, strokeWidth: 1.6 }} />
        </span>
        <span className="nm">{item.name}</span>
      </div>
      {item.is_featured && <span className="tag">★ Featured</span>}
      {item.description && <div className="desc">{item.description}</div>}
      <div className="foot">
        <span>{item.category || item.source}</span>
        <span className="dl">
          <Download
            style={{
              width: 9,
              height: 9,
              marginRight: 3,
              verticalAlign: '-1px',
            }}
          />
          {item.install_count ?? item.use_count ?? 0}
        </span>
      </div>
    </button>
  )
}
