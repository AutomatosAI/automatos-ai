'use client'

/**
 * PRD-255 US-007 — the type scale on the Brand kit page: each step's size, line height and
 * weight, with a live sample line in the kit's fonts (the headings' steps in the heading
 * font). One default scale for every kit (Decision Q5); the server bounds each value.
 */
import type { BrandKit, BrandTypeStep, BrandTypeStepName } from '@/components/documents/blocks/types'
import { NumberField, SectionTitle } from './brand-kit-inputs'

export const TYPE_STEP_LABELS: Record<BrandTypeStepName, string> = {
  display: 'Display',
  h1: 'Heading 1',
  h2: 'Heading 2',
  h3: 'Heading 3',
  body: 'Body',
  small: 'Small',
  caption: 'Caption',
}
const HEADING_STEPS: ReadonlySet<BrandTypeStepName> = new Set(['display', 'h1', 'h2', 'h3'])
const SIZE_STEP_PT = 0.5
const WEIGHT_STEP = 100
const SAMPLE_TEXT = 'The quick brown fox jumps over the lazy dog'

interface BrandKitTypeProps {
  kit: BrandKit
  patch: (p: Partial<BrandKit>) => void
}

function sampleFont(kit: BrandKit, step: BrandTypeStepName): string | undefined {
  const heading = HEADING_STEPS.has(step) ? kit.heading_font : ''
  return heading || kit.font_family || undefined
}

export function BrandKitType({ kit, patch }: BrandKitTypeProps) {
  const scale = kit.type_scale
  if (!scale) return null
  const setStep = (step: BrandTypeStepName, change: Partial<BrandTypeStep>) =>
    patch({ type_scale: { ...scale, [step]: { ...scale[step], ...change } } })
  return (
    <section aria-label="Type">
      <SectionTitle title="Type">One scale for every document: each step&apos;s size, line height and weight.</SectionTitle>
      <div className="space-y-2">
        {(Object.keys(TYPE_STEP_LABELS) as BrandTypeStepName[]).map((step) => (
          <div key={step} className="rounded-md border p-2" data-testid={`type-${step}`}>
            <div className="grid grid-cols-3 gap-2 sm:grid-cols-[8rem_repeat(3,minmax(0,1fr))]">
              <p className="col-span-3 self-center text-xs font-medium sm:col-span-1">{TYPE_STEP_LABELS[step]}</p>
              <NumberField id={`type-${step}-size`} label="Size" unit="pt" step={SIZE_STEP_PT} value={scale[step].size_pt} onChange={(size_pt) => setStep(step, { size_pt })} />
              <NumberField id={`type-${step}-line`} label="Line" unit="pt" step={SIZE_STEP_PT} value={scale[step].line_pt} onChange={(line_pt) => setStep(step, { line_pt })} />
              <NumberField id={`type-${step}-weight`} label="Weight" step={WEIGHT_STEP} value={scale[step].weight} onChange={(weight) => setStep(step, { weight })} />
            </div>
            <p
              className="mt-1 overflow-hidden text-ellipsis whitespace-nowrap"
              data-testid={`type-sample-${step}`}
              style={{
                fontSize: `${scale[step].size_pt}pt`, lineHeight: `${scale[step].line_pt}pt`,
                fontWeight: scale[step].weight, fontFamily: sampleFont(kit, step),
              }}
            >
              {SAMPLE_TEXT}
            </p>
          </div>
        ))}
      </div>
    </section>
  )
}
