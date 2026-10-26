/**
 * AiVoiceLabel — "AI voice assistant", shown for the whole of an Auto Live call
 * (PRD-264 EUAI-04, EU AI Act Art. 50(1) and 50(5)).
 *
 * The caller also hears it: the call opens with Auto saying it is an AI, once
 * (orchestrator/modules/voice/ai_notice.py). On screen it is a tiny muted
 * caption, never a pill or a badge, and no control waits on it (the product
 * owner's policy, 23 Oct 2026). Theme tokens only (the `dark:` variant is inert
 * in this frontend), and screen readers get the same words. It takes no props,
 * so no host can reword or hide it.
 */

export const AI_VOICE_LABEL = 'AI voice assistant'

export function AiVoiceLabel() {
  return (
    <span
      role="note"
      aria-label={AI_VOICE_LABEL}
      data-testid="ai-voice-label"
      className="shrink-0 text-[10px] leading-none text-muted-foreground"
    >
      {AI_VOICE_LABEL}
    </span>
  )
}
