'use client'

import { useMemo } from 'react'
import tooltipsData from './tooltips.json'

/**
 * Tooltip data structure
 */
export interface TooltipData {
  text: string
  docLink?: string
  example?: string
}

/**
 * Hook for accessing tooltip data
 * 
 * @param id - Dot-notation path to tooltip (e.g., "agents.roster.create_button")
 * @returns Tooltip data object or null if not found
 * 
 * @example
 * ```tsx
 * const tooltip = useTooltip('agents.roster.create_button')
 * 
 * if (tooltip) {
 *   <HelpTooltip 
 *     text={tooltip.text}
 *     docLink={tooltip.docLink}
 *   />
 * }
 * ```
 */
export function useTooltip(id: string): TooltipData | null {
  return useMemo(() => {
    const parts = id.split('.')
    let current: any = tooltipsData

    // Traverse nested object using dot notation
    for (const part of parts) {
      if (current && typeof current === 'object' && part in current) {
        current = current[part]
      } else {
        console.warn(`Tooltip not found: ${id}`)
        return null
      }
    }

    // Return if it's a tooltip object (has "text" property)
    if (current && typeof current === 'object' && 'text' in current) {
      return current as TooltipData
    }

    console.warn(`Tooltip data invalid for: ${id}`)
    return null
  }, [id])
}

/**
 * Hook for accessing multiple tooltips at once
 * 
 * @param ids - Array of tooltip IDs
 * @returns Map of ID to tooltip data
 * 
 * @example
 * ```tsx
 * const tooltips = useTooltips([
 *   'agents.roster.create_button',
 *   'agents.roster.search',
 *   'agents.roster.filter'
 * ])
 * 
 * <HelpTooltip {...tooltips['agents.roster.create_button']} />
 * ```
 */
export function useTooltips(ids: string[]): Record<string, TooltipData | null> {
  return useMemo(() => {
    const result: Record<string, TooltipData | null> = {}
    
    for (const id of ids) {
      result[id] = useTooltip(id)
    }
    
    return result
  }, [ids.join(',')])
}

/**
 * Hook for getting all tooltips for a specific page
 * 
 * @param page - Page name (e.g., "agents", "workflows", "knowledge")
 * @returns All tooltips for that page
 * 
 * @example
 * ```tsx
 * const agentTooltips = usePageTooltips('agents')
 * // Returns all tooltips under "agents" key
 * ```
 */
export function usePageTooltips(page: string): Record<string, any> | null {
  return useMemo(() => {
    if (page in tooltipsData) {
      return (tooltipsData as any)[page]
    }
    
    console.warn(`No tooltips found for page: ${page}`)
    return null
  }, [page])
}

/**
 * Utility function to check if tooltip exists
 * 
 * @param id - Tooltip ID
 * @returns True if tooltip exists
 */
export function hasTooltip(id: string): boolean {
  const parts = id.split('.')
  let current: any = tooltipsData

  for (const part of parts) {
    if (current && typeof current === 'object' && part in current) {
      current = current[part]
    } else {
      return false
    }
  }

  return current && typeof current === 'object' && 'text' in current
}

/**
 * Get all available tooltip IDs (for debugging/documentation)
 * 
 * @returns Array of all tooltip paths
 */
export function getAllTooltipIds(): string[] {
  const ids: string[] = []

  function traverse(obj: any, path: string = '') {
    for (const key in obj) {
      const newPath = path ? `${path}.${key}` : key
      const value = obj[key]

      if (value && typeof value === 'object') {
        if ('text' in value) {
          // This is a tooltip
          ids.push(newPath)
        } else {
          // This is a category, traverse deeper
          traverse(value, newPath)
        }
      }
    }
  }

  traverse(tooltipsData)
  return ids
}

/**
 * Get tooltip count (for documentation/stats)
 * 
 * @returns Total number of tooltips
 */
export function getTooltipCount(): number {
  return getAllTooltipIds().length
}

