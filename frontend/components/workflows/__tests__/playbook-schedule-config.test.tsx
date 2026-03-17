import { render, screen, fireEvent } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { useForm, FormProvider } from 'react-hook-form'
import { PlaybookScheduleConfig } from '../playbook-schedule-config'
import type { PlaybookFormValues } from '../create-playbook-modal'
import { describe, it, expect, vi } from 'vitest'

// Wrapper that provides react-hook-form context with default schedule_config
function Wrapper({
  defaultType = 'manual',
  cronExpression = '',
  webhookId,
  children,
}: {
  defaultType?: string
  cronExpression?: string
  webhookId?: string
  children?: React.ReactNode
}) {
  const methods = useForm<PlaybookFormValues>({
    defaultValues: {
      name: 'Test Playbook',
      description: 'Test',
      inputs: '{}',
      outputs: '{}',
      steps: [],
      execution_config: {
        mode: 'sequential',
        max_retries: 1,
        timeout_per_step: 120000,
        total_timeout: 600000,
        auto_learning: true,
        parallel_limit: 5,
        memory_isolation: 'shared',
      },
      schedule_config: {
        type: defaultType,
        cron_expression: cronExpression,
        trigger_config: {},
      },
    },
  })

  return (
    <FormProvider {...methods}>
      <PlaybookScheduleConfig webhookId={webhookId} />
    </FormProvider>
  )
}

// ---------------------------------------------------------------------------
// Schedule type buttons
// ---------------------------------------------------------------------------

describe('PlaybookScheduleConfig', () => {
  it('renders all three schedule types', () => {
    render(<Wrapper />)

    expect(screen.getByText('Manual')).toBeInTheDocument()
    expect(screen.getByText('Scheduled')).toBeInTheDocument()
    expect(screen.getByText('Triggered')).toBeInTheDocument()
  })

  it('manual is default', () => {
    render(<Wrapper />)

    // Manual section content should be visible
    expect(screen.getByText(/executed manually/i)).toBeInTheDocument()
  })

  // ---------------------------------------------------------------------------
  // Cron UI
  // ---------------------------------------------------------------------------

  it('clicking Scheduled shows cron UI', async () => {
    render(<Wrapper />)

    const scheduledBtn = screen.getByText('Scheduled')
    await userEvent.click(scheduledBtn)

    // Quick Picks dropdown should appear
    expect(screen.getByText('Quick Picks')).toBeInTheDocument()
    // Cron Expression input should appear
    expect(screen.getByText('Cron Expression')).toBeInTheDocument()
  })

  it('quick pick sets cron expression', async () => {
    render(<Wrapper defaultType="cron" />)

    // Select "Daily at 9am" from the dropdown
    const select = screen.getByRole('combobox') as HTMLSelectElement
    await userEvent.selectOptions(select, '0 9 * * *')

    // The cron input should now have the value
    const cronInput = screen.getByPlaceholderText('0 9 * * 1-5') as HTMLInputElement
    expect(cronInput.value).toBe('0 9 * * *')
  })

  it('valid cron shows next runs preview', () => {
    render(<Wrapper defaultType="cron" cronExpression="0 9 * * *" />)

    // "Next 5 runs" section should appear
    expect(screen.getByText('Next 5 runs')).toBeInTheDocument()
  })

  it('invalid cron shows error message', () => {
    render(<Wrapper defaultType="cron" cronExpression="not-valid" />)

    expect(screen.getByText(/invalid cron/i)).toBeInTheDocument()
  })

  // ---------------------------------------------------------------------------
  // Trigger/Webhook UI
  // ---------------------------------------------------------------------------

  it('clicking Triggered shows webhook UI', async () => {
    render(<Wrapper />)

    const triggerBtn = screen.getByText('Triggered')
    await userEvent.click(triggerBtn)

    // Webhook URL section header should appear
    expect(screen.getByText('Webhook URL')).toBeInTheDocument()
  })

  it('webhook URL shown when webhookId provided', async () => {
    render(<Wrapper defaultType="trigger" webhookId="abc123" />)

    // Should render the full webhook URL
    const urlInput = screen.getByDisplayValue(/\/api\/webhooks\/recipe\/abc123/) as HTMLInputElement
    expect(urlInput).toBeInTheDocument()
  })
})
