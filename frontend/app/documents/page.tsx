'use client'

import { MainLayout } from '@/components/layout/main-layout'
import { DocumentManagement } from "@/components/documents/document-management"
import { usePageAPI } from '@/hooks/use-page-api'

export default function DocumentsPage() {
  usePageAPI('documents')
  
  return (
    <MainLayout>
      <DocumentManagement />
    </MainLayout>
  )
}
