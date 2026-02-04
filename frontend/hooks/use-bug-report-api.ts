import { useMutation } from '@tanstack/react-query'
import { toast } from 'react-hot-toast'
import apiClient from '../lib/api-client'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface BugReportContext {
  url?: string
  page?: string
  user_agent?: string
  viewport?: string
  user_email?: string
  user_name?: string
  console_errors?: string[]
  timestamp?: string
}

export interface BugReportRequest {
  title: string
  description: string
  severity: string
  category: string
  screenshot_base64?: string
  context: BugReportContext
}

export interface BugReportResponse {
  success: boolean
  jira_key?: string
  jira_url?: string
  message: string
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useSubmitBugReport() {
  return useMutation({
    mutationFn: (data: BugReportRequest) =>
      apiClient.post<BugReportResponse>('/api/bug-reports/', data),
    onSuccess: (data) => {
      if (data.success) {
        toast.success(data.jira_key ? `Bug report created: ${data.jira_key}` : 'Bug report submitted')
      } else {
        toast.error(data.message || 'Failed to submit bug report')
      }
    },
    onError: (error: any) => {
      const msg = error?.message || 'Failed to submit bug report'
      toast.error(msg)
    },
  })
}
