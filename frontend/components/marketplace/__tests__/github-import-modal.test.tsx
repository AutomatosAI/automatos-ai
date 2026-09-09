/**
 * A fresh install has no skills. The import modal must point at the free
 * baseline library and fill the URL in one click, and honour a prefill from a
 * caller (the Skills tab opens it on the baseline repo).
 */
import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'

vi.mock('@/lib/api-client', () => ({ apiClient: { post: vi.fn() } }))
vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }))

import { GitHubImportModal } from '../github-import-modal'
import { BASELINE_SKILLS_REPO_URL } from '@/lib/baseline-skills'

const PLACEHOLDER = 'https://github.com/user/repo'

describe('GitHubImportModal — baseline skills', () => {
  it('offers the baseline repo and fills the URL on click', () => {
    render(<GitHubImportModal open onClose={() => {}} onImportComplete={() => {}} />)
    expect(screen.getByText(/Start with the free baseline skills/)).toBeInTheDocument()
    expect(screen.getByPlaceholderText(PLACEHOLDER)).toHaveValue('')
    fireEvent.click(screen.getByRole('button', { name: /use baseline repo/i }))
    expect(screen.getByPlaceholderText(PLACEHOLDER)).toHaveValue(BASELINE_SKILLS_REPO_URL)
  })

  it('prefills from initialUrl', () => {
    render(
      <GitHubImportModal open onClose={() => {}} onImportComplete={() => {}} initialUrl={BASELINE_SKILLS_REPO_URL} />,
    )
    expect(screen.getByPlaceholderText(PLACEHOLDER)).toHaveValue(BASELINE_SKILLS_REPO_URL)
  })

  it('the baseline URL is the public skills repo', () => {
    expect(BASELINE_SKILLS_REPO_URL).toBe('https://github.com/AutomatosAI/automatos-skills.git')
  })
})
