import { describe, it, expect } from 'vitest'
import { autoPrompt, emailPrompt, playbookStepJson, schedulePrompt } from '../promptSnippets'

const t = { id: '11111111-1111-1111-1111-111111111111', name: 'Weekly Report', format: 'pdf', data_fields: ['title', 'summary'] }

describe('promptSnippets', () => {
  it('names the template by id and lists the fill-in fields', () => {
    const p = autoPrompt(t, 'AI news this week')
    expect(p).toContain('Research AI news this week')
    expect(p).toContain('"Weekly Report" template (template_id 11111111-1111-1111-1111-111111111111)')
    expect(p).toContain('Fill these fields from your research: title, summary.')
    expect(p).toContain('Save it to Deliverables')
  })
  it('says when a template has no fields', () => {
    expect(autoPrompt({ ...t, data_fields: [] })).toContain('no fill-in fields')
  })
  it('prefixes the schedule and appends the email delivery', () => {
    expect(schedulePrompt(t, 'x', 'every Friday at 17:00')).toMatch(/^Every Friday at 17:00: Research x/)
    expect(emailPrompt(t, 'x', 'marketing@acme.com')).toContain('email the share link to marketing@acme.com')
  })
  it('explains list fields as rows with keys (PRD-243)', () => {
    const inv = { ...t, name: 'Branded Invoice', data_fields: ['client_name', 'line_items'], list_fields: [{ field: 'line_items', columns: ['description', 'quantity', 'total'] }] }
    expect(autoPrompt(inv, 'x')).toContain('line_items is a list of rows, each with description, quantity, total.')
    expect(JSON.parse(playbookStepJson(inv)).data).toEqual({ client_name: '{{step_1.output}}', line_items: [] })
  })
  it('emits a valid generate_document playbook step', () => {
    const step = JSON.parse(playbookStepJson(t))
    expect(step).toEqual({
      type: 'generate_document',
      title: 'Weekly Report',
      format: 'pdf',
      template_id: t.id,
      data: { title: '{{step_1.output}}', summary: '{{step_1.output}}' },
    })
  })
})
