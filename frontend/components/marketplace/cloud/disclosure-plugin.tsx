'use client'

/**
 * PRD-270 W2 US-270-14c (§5.6, K15, K18) — a Cloud plugin's part of the disclosure: Automatos Cloud's scan
 * report as the Sync kept it, this install's own static check of its files, what it contains, the skills it
 * added here, and that an update applies to every workspace that enabled it (a plugin is enabled by
 * reference, so the row they run moves with it). Every text is the server's, rendered as text.
 */
import type { PluginCloudReport, PluginDisclosure, PluginLocalCheck } from '@/lib/marketplace/cloud-items'

import { Line, Lines, Section, TextBlock } from './disclosure-parts'
import { NO_REPORT, UPDATES_IN_PLACE, contentsLine, findingLine, riskLine, statusLine } from './item-copy'

function CloudReport({ scan }: { scan: PluginCloudReport | null }) {
  if (!scan) return <p className="text-sm text-muted-foreground">{NO_REPORT}</p>
  const facts = [statusLine('Verdict', scan.verdict), statusLine('Review', scan.status), riskLine(scan.riskScore)]
  return (
    <div className="space-y-2">
      <p className="text-sm text-foreground">{facts.filter(Boolean).join(' · ')}</p>
      {scan.summary && <TextBlock text={scan.summary} />}
      <Lines>{scan.findings.map((finding, index) => (
        <Line key={index} text={findingLine(finding.severity, finding.description, null, null)} />
      ))}</Lines>
    </div>
  )
}

function LocalCheck({ check }: { check: PluginLocalCheck | null }) {
  if (!check) return <p className="text-sm text-muted-foreground">{NO_REPORT}</p>
  return (
    <div className="space-y-2">
      <p className="text-sm text-foreground">{statusLine('Result', check.status) ?? NO_REPORT}</p>
      <Lines>{check.findings.map((finding, index) => (
        <Line key={index} text={findingLine(finding.severity, finding.description, finding.file, finding.line)} />
      ))}</Lines>
    </div>
  )
}

export function PluginSections({ disclosure }: { disclosure: PluginDisclosure }) {
  return (
    <>
      <Section title="Automatos Cloud's scan report"><CloudReport scan={disclosure.scan} /></Section>
      <Section title="This install's check"><LocalCheck check={disclosure.localCheck} /></Section>
      <Section title="Contents">
        <p className="text-sm text-foreground">{contentsLine(disclosure.contents)}</p>
      </Section>
      <Section title="Skills it adds">
        <Lines>{disclosure.skills.map((name, index) => <Line key={index} text={name} />)}</Lines>
      </Section>
      <p className="text-sm text-muted-foreground">{UPDATES_IN_PLACE}</p>
    </>
  )
}
