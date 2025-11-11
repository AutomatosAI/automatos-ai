/**
 * AI Model definitions and configurations
 */

import type { Model } from '@/types'

export const DEFAULT_CHAT_MODEL = 'gpt-4'

export const chatModels: Model[] = [
  {
    id: 'gpt-4',
    name: 'GPT-4',
    provider: 'openai',
    description: 'Most capable model for complex tasks',
    contextWindow: 8192,
    pricing: {
      input: 0.03,
      output: 0.06,
    },
  },
  {
    id: 'gpt-4-turbo',
    name: 'GPT-4 Turbo',
    provider: 'openai',
    description: 'Faster GPT-4 with larger context window',
    contextWindow: 128000,
    pricing: {
      input: 0.01,
      output: 0.03,
    },
  },
  {
    id: 'gpt-3.5-turbo',
    name: 'GPT-3.5 Turbo',
    provider: 'openai',
    description: 'Fast and cost-effective for simple tasks',
    contextWindow: 16384,
    pricing: {
      input: 0.0005,
      output: 0.0015,
    },
  },
  {
    id: 'claude-3-opus',
    name: 'Claude 3 Opus',
    provider: 'anthropic',
    description: 'Most capable Anthropic model',
    contextWindow: 200000,
    pricing: {
      input: 0.015,
      output: 0.075,
    },
  },
  {
    id: 'claude-3-sonnet',
    name: 'Claude 3 Sonnet',
    provider: 'anthropic',
    description: 'Balanced performance and speed',
    contextWindow: 200000,
    pricing: {
      input: 0.003,
      output: 0.015,
    },
  },
]

/**
 * Get model by ID
 */
export function getModelById(id: string): Model | undefined {
  return chatModels.find(m => m.id === id)
}

/**
 * Get models by provider
 */
export function getModelsByProvider(provider: string): Model[] {
  return chatModels.filter(m => m.provider === provider)
}

