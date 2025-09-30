
'use client'

import { FieldVisualization } from '@/components/field-theory/field-visualization'
import { usePageAPI } from '@/hooks/use-page-api'

export default function FieldTheoryPage() {
  usePageAPI('field-theory')
  
  return <FieldVisualization />
}
