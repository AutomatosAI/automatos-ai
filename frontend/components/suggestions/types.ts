/**
 * Type definitions for Dynamic Tool Suggestions (PRD-40)
 *
 * Shared types used across suggestion components.
 */

/**
 * API response from GET /api/tools/{app_name}/suggestions
 */
export interface SuggestionResponse {
  /** The app name (uppercase, e.g., "GMAIL") */
  app: string
  /** Array of suggestion prompt strings */
  suggestions: string[]
  /** Source of suggestions: "curated" or "generated" */
  source: 'curated' | 'generated'
}

/**
 * Props for ToolSuggestionBar component
 */
export interface ToolSuggestionBarProps {
  /** Array of suggestion text strings to display */
  suggestions: string[]
  /** Name of the active tool (e.g., "GMAIL", "SLACK") */
  activeTool: string | null
  /** Callback when a suggestion is clicked */
  onSuggestionClick: (suggestion: string) => void
  /** Callback when close button is clicked */
  onClose: () => void
  /** Optional loading state */
  isLoading?: boolean
  /** Optional className for custom styling */
  className?: string
}

/**
 * Props for SuggestionChip component
 */
export interface SuggestionChipProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  /** The suggestion text to display */
  text: string
}
