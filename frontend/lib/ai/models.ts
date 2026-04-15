/**
 * AI Model definitions and configurations
 */

import type { Model } from '@/types'

import { LLM_DEFAULTS } from '@/lib/llm-defaults'
export { LLM_DEFAULTS }
export const DEFAULT_CHAT_MODEL = LLM_DEFAULTS.model_id

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
  {
    id: 'grok-2-latest',
    name: 'Grok 2',
    provider: 'grok',
    description: 'xAI flagship model with real-time knowledge',
    contextWindow: 131072,
    pricing: {
      input: 0.002,
      output: 0.010,
    },
  },
  {
    id: 'grok-2-1212',
    name: 'Grok 2 (Dec 2024)',
    provider: 'grok',
    description: 'Grok 2 December 2024 release',
    contextWindow: 131072,
    pricing: {
      input: 0.002,
      output: 0.010,
    },
  },
  {
    id: 'grok-3-beta',
    name: 'Grok 3 Beta',
    provider: 'grok',
    description: 'Next-gen Grok with enhanced reasoning',
    contextWindow: 131072,
    pricing: {
      input: 0.005,
      output: 0.015,
    },
  },
  {
    id: 'google/gemma-2-9b-it',
    name: 'Gemma 2 9B',
    provider: 'huggingface',
    description: 'Google Gemma 2 instruction-tuned',
    contextWindow: 8192,
    pricing: { input: 0.0, output: 0.0 },
  },
  {
    id: 'microsoft/Phi-3.5-mini-instruct',
    name: 'Phi 3.5 Mini',
    provider: 'huggingface',
    description: 'Microsoft compact reasoning model',
    contextWindow: 4096,
    pricing: { input: 0.0, output: 0.0 },
  },
  {
    id: 'Qwen/Qwen2.5-7B-Instruct',
    name: 'Qwen 2.5 7B',
    provider: 'huggingface',
    description: 'Alibaba Qwen 2.5 instruction model',
    contextWindow: 8192,
    pricing: { input: 0.0, output: 0.0 },
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

