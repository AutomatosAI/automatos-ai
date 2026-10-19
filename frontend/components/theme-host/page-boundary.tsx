'use client'

import { unstable_rethrow } from 'next/navigation'
import React from 'react'

import { ErrorState } from '@/components/shared/error-state'

/** Marks the page's error UI inside a theme's frame, for tests and the browser lane. */
export const PAGE_ERROR_TEST_ID = 'theme-page-error'
export const PAGE_ERROR_TITLE = 'This page could not be shown'
export const PAGE_ERROR_TEXT = 'Something on this page failed. The rest of the app still works.'

const LOG_PREFIX = '[theme-host]'

interface Props {
  children: React.ReactNode
}

interface State {
  hasError: boolean
}

/**
 * PRD-261 A-04a (A0 plan §7 Q7) — the page inside a theme's frame gets its own boundary. A page
 * that throws is the page's failure, not the theme's: the core's error UI draws in the page's
 * place, the frame stays, and the host is never told (no reportRenderFailure), so a broken page
 * never takes the theme down or counts toward the three-failure revert (A-04c). The error goes to
 * the console only; the UI shows a plain message, never the error's text. Next's own control-flow
 * errors (notFound(), redirect(), a bailout to client rendering) are not the page's failure: they
 * are rethrown to Next's boundaries above, as Next's own error boundary does.
 */
export class PageBoundary extends React.Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = { hasError: false }
    this.retry = this.retry.bind(this)
  }

  static getDerivedStateFromError(error: unknown): State {
    unstable_rethrow(error)
    return { hasError: true }
  }

  componentDidCatch(error: unknown) {
    console.error(LOG_PREFIX, 'page', error)
  }

  retry() {
    this.setState({ hasError: false })
  }

  render() {
    if (!this.state.hasError) return this.props.children
    return (
      <div data-testid={PAGE_ERROR_TEST_ID}>
        <ErrorState title={PAGE_ERROR_TITLE} description={PAGE_ERROR_TEXT} onRetry={this.retry} />
      </div>
    )
  }
}
