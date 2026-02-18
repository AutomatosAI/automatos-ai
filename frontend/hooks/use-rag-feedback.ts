'use client'

import { useState } from 'react'
import { apiClient } from '@/lib/api-client'
import { toast } from 'sonner'

interface FeedbackPayload {
  query: string
  response_text?: string
  chunk_ids?: number[]
  document_ids?: number[]
  rating?: number
  feedback_type: 'thumbs_up' | 'thumbs_down' | 'star_rating' | 'correction'
  correction_text?: string
  execution_time_ms?: number
}

export function useRAGFeedback() {
  const [isSubmitting, setIsSubmitting] = useState(false)

  const submitFeedback = async (payload: FeedbackPayload) => {
    setIsSubmitting(true)
    try {
      const res = await apiClient.post('/api/rag/feedback', payload)
      return res
    } catch (err) {
      toast.error('Failed to submit feedback')
      throw err
    } finally {
      setIsSubmitting(false)
    }
  }

  const thumbsUp = (query: string, opts?: Partial<FeedbackPayload>) =>
    submitFeedback({ query, feedback_type: 'thumbs_up', ...opts })

  const thumbsDown = (query: string, opts?: Partial<FeedbackPayload>) =>
    submitFeedback({ query, feedback_type: 'thumbs_down', ...opts })

  const submitCorrection = (query: string, correction: string, opts?: Partial<FeedbackPayload>) =>
    submitFeedback({ query, feedback_type: 'correction', correction_text: correction, ...opts })

  const submitRating = (query: string, rating: number, opts?: Partial<FeedbackPayload>) =>
    submitFeedback({ query, feedback_type: 'star_rating', rating, ...opts })

  return { submitFeedback, thumbsUp, thumbsDown, submitCorrection, submitRating, isSubmitting }
}
