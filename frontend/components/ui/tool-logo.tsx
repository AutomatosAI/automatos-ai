/**
 * Logo Component - Display Real Company Logos
 * ============================================
 * 
 * A reusable component for displaying tool/credential logos.
 * Logo path is stored in the database metadata.
 * 
 * Usage:
 * ```tsx
 * <ToolLogo logo="/logos/Discord.png" name="Discord Bot MCP" size={32} />
 * ```
 */

import React, { useState } from 'react'
import { LucideIcon } from 'lucide-react'
import * as LucideIcons from 'lucide-react'

interface ToolLogoProps {
    /** Logo path from metadata (e.g. "/logos/Discord.png") */
    logo?: string
    /** Tool or credential name (for fallback) */
    name: string
    /** Logo size in pixels */
    size?: number
    /** Fallback icon name from lucide-react */
    fallbackIcon?: string
    /** Additional CSS classes */
    className?: string
    /** Show background */
    showBackground?: boolean
}

export function ToolLogo({
    logo,
    name,
    size = 32,
    fallbackIcon,
    className = '',
    showBackground = true
}: ToolLogoProps) {
    const [imageError, setImageError] = useState(false)

    // If logo exists and hasn't errored, show it
    if (logo && !imageError) {
        return (
            <div
                className={`flex items-center justify-center ${showBackground ? 'bg-secondary/10' : ''} rounded-lg ${className}`}
                style={{
                    width: size,
                    height: size,
                }}
            >
                <img
                    src={logo}
                    alt={`${name} logo`}
                    className="w-full h-full object-contain p-1"
                    onError={() => setImageError(true)}
                />
            </div>
        )
    }

    // Fallback to icon or emoji
    const FallbackIcon = fallbackIcon ? (LucideIcons as any)[fallbackIcon] : null

    return (
        <div
            className={`inline-flex items-center justify-center ${showBackground ? 'bg-white/10 rounded-lg' : ''} ${className}`}
            style={{
                width: size,
                height: size,
                minWidth: size,
                minHeight: size
            }}
        >
            {FallbackIcon ? (
                <FallbackIcon
                    className="text-muted-foreground"
                    size={size * 0.6}
                />
            ) : (
                <span className="text-lg">
                    {getEmojiForTool(name)}
                </span>
            )}
        </div>
    )
}

/**
 * Get emoji for a tool based on its name
 */
function getEmojiForTool(name: string): string {
    const lowerName = name.toLowerCase()

    // Communication
    if (lowerName.includes('slack')) return '💬'
    if (lowerName.includes('discord')) return '🎮'
    if (lowerName.includes('telegram')) return '✈️'
    if (lowerName.includes('email') || lowerName.includes('mail')) return '📧'

    // Development
    if (lowerName.includes('github') || lowerName.includes('git')) return '🐙'
    if (lowerName.includes('gitlab')) return '🦊'
    if (lowerName.includes('docker')) return '🐳'
    if (lowerName.includes('kubernetes')) return '☸️'

    // Cloud
    if (lowerName.includes('aws') || lowerName.includes('amazon')) return '☁️'
    if (lowerName.includes('azure') || lowerName.includes('microsoft')) return '🔷'
    if (lowerName.includes('google') || lowerName.includes('gcp')) return '🔍'

    // Database
    if (lowerName.includes('mongo')) return '🍃'
    if (lowerName.includes('postgres') || lowerName.includes('sql')) return '🐘'
    if (lowerName.includes('redis')) return '🔴'

    // AI
    if (lowerName.includes('openai') || lowerName.includes('gpt')) return '🤖'
    if (lowerName.includes('anthropic') || lowerName.includes('claude')) return '🧠'

    // Default
    return '🔧'
}
