/**
 * AiVoiceLabel — "AI voice assistant", shown for the whole of an Auto Live call
 * (PRD-264 EUAI-04, EU AI Act Art. 50(1) and 50(5)).
 *
 * The caller also hears it: the call opens with Auto saying it is an AI, and
 * reminds them later (orchestrator/modules/voice/ai_notice.py). On screen the
 * label is persistent and perceivable in both themes: theme tokens only (the
 * `dark:` variant is inert in this frontend), and screen readers get the same
 * words. It takes no props, so no host can reword or hide it.
 */

export const AI_VOICE_LABEL = 'AI voice assistant'

export function AiVoiceLabel() {
  return (
    <span
      role="note"
      aria-label={AI_VOICE_LABEL}
      data-testid="ai-voice-label"
      className="inline-flex shrink-0 items-center rounded-full border border-border bg-muted px-2 py-0.5 text-[10px] font-medium leading-none text-foreground"
    >
      {AI_VOICE_LABEL}
    </span>
  )
}
