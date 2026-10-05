'use client'

/** The charts PandasAI drew for a text artifact, each with Download and Copy. */
import { toast } from 'sonner'

import type { PandasAIChart } from '@/types/chat'

function chartDataUrl(chart: PandasAIChart): string {
  return `data:${chart.mime_type};base64,${chart.base64}`
}

function downloadChart(chart: PandasAIChart) {
  const link = document.createElement('a')
  link.href = chartDataUrl(chart)
  link.download = chart.filename
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
}

async function copyChart(chart: PandasAIChart) {
  if (!navigator.clipboard) {
    toast.error('Clipboard API is not available')
    return
  }
  try {
    await navigator.clipboard.writeText(chartDataUrl(chart))
    toast.success('Copied to clipboard')
  } catch {
    toast.error('Failed to copy to clipboard')
  }
}

const CHART_BUTTON =
  'rounded border border-border/60 px-2 py-1 text-[11px] uppercase tracking-wide text-foreground/90 hover:border-primary/60 hover:text-primary/80'

export function PandasAICharts({ charts }: { charts: PandasAIChart[] }) {
  return (
    <div className="space-y-4">
      <h4 className="text-sm font-semibold text-foreground/90 uppercase tracking-wide">
        PandasAI Charts
      </h4>
      <div className="grid gap-4 md:grid-cols-2">
        {charts.map((chart, idx) => (
          <div
            key={`${chart.filename}-${idx}`}
            className="rounded-lg border border-gray-800/60 bg-background/40 p-4 flex flex-col items-center gap-3"
          >
            <img
              src={chartDataUrl(chart)}
              alt={chart.filename}
              className="rounded-md border border-gray-800/40 max-h-72 w-full object-contain"
            />
            <div className="flex w-full items-center justify-between text-xs text-muted-foreground">
              <span className="truncate">{chart.filename}</span>
              <div className="flex items-center gap-2">
                <button className={CHART_BUTTON} onClick={() => downloadChart(chart)}>
                  Download
                </button>
                <button className={CHART_BUTTON} onClick={() => copyChart(chart)}>
                  Copy
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
