// Copy-ready instructions that make a template USABLE from chat, a schedule, or a
// playbook (PRD-242 S5). The text names the template by id — the id is what
// generate_document(template_id=…) takes, so an agent cannot pick the wrong one.
import type { TemplateSummary } from './types'

type TemplateRef = Pick<TemplateSummary, 'id' | 'name' | 'format' | 'data_fields'>

function fieldsClause(t: TemplateRef): string {
  if (!t.data_fields.length) return 'The template has no fill-in fields.'
  return `Fill these fields from your research: ${t.data_fields.join(', ')}.`
}

function fmt(t: TemplateRef): string {
  return String(t.format || 'pdf').toUpperCase()
}

// Ask Auto (or any agent) in chat.
export function autoPrompt(t: TemplateRef, topic = '<what to research>'): string {
  return [
    `Research ${topic}, then generate a ${fmt(t)} with the "${t.name}" template (template_id ${t.id}).`,
    fieldsClause(t),
    'Save it to Deliverables and give me the link.',
  ].join(' ')
}

// Same, on a schedule — the agent files it via platform_schedule_task.
export function schedulePrompt(t: TemplateRef, topic = '<what to research>', when = 'every Monday at 09:00'): string {
  return `${when.charAt(0).toUpperCase()}${when.slice(1)}: ${autoPrompt(t, topic)}`
}

// Deliver by email — the tool result carries a no-sign-in share link (valid 7 days).
export function emailPrompt(t: TemplateRef, topic = '<what to research>', recipient = 'marketing@yourcompany.com'): string {
  return `${autoPrompt(t, topic)} Then email the share link to ${recipient} with a two-line summary.`
}

// A deterministic playbook step (recipe_executor generate_document step type).
export function playbookStepJson(t: TemplateRef): string {
  const data = Object.fromEntries(t.data_fields.map((f) => [f, `{{step_1.output}}`]))
  return JSON.stringify(
    {
      type: 'generate_document',
      title: t.name,
      format: t.format || 'pdf',
      template_id: t.id,
      data,
    },
    null,
    2,
  )
}
