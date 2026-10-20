'use client'

import { MultimodalInput, type MultimodalInputProps } from '@/components/chatbot/multimodal-input'

import { AiLabelFence } from './ai-label-fence'

/**
 * PRD-264 EUAI-11a — the default chat's input (Mission Mode's too) inside the AI label's fence:
 * disabled while a theme hides the host's AI label. components/chatbot/chat.tsx imports this in
 * place of MultimodalInput, as it imports useChatWithReceipts in place of useChat: both files are
 * far over the code-shape limits, so neither is edited beyond that import.
 */
export function FencedMultimodalInput(props: MultimodalInputProps) {
  return (
    <AiLabelFence>
      <MultimodalInput {...props} />
    </AiLabelFence>
  )
}
