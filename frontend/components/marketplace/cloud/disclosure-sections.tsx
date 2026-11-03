'use client'

/**
 * PRD-270 W2 US-270-14c (§5.6, K18, K30) — what a Cloud item will use, per kind, from the local disclosure
 * read (`GET /api/marketplace/cloud-sync/items/{kind}/{key}/disclosure`, US-270-14a):
 *
 * - agent: its prompt (or "No custom prompt"), its model and whether this install has it, its apps with
 *   "needs a connection", its skills and plugins with "not on this install";
 * - Playbook: each step's helper (or "needs a helper") and prompt, and the agents its install adds;
 * - package: its members, the connections its setup asks for, each agent member's prompt;
 * - plugin: Automatos Cloud's scan report, this install's static check, its contents and skills, and that an
 *   update applies to every workspace that enabled it (disclosure-plugin.tsx);
 * - for every kind, what didn't resolve here as "Needs …" lines.
 *
 * Every text is the server's and is rendered as text: a prompt in a whitespace-pre-wrap block, never as
 * markdown or HTML.
 */
import type {
  AgentDisclosure, CloudDisclosure, CloudPrompt, CloudUnresolved, PackageDisclosure, PlaybookDisclosure, PlaybookStep,
} from '@/lib/marketplace/cloud-items'

import { Line, Lines, Note, Section, TextBlock } from './disclosure-parts'
import { PluginSections } from './disclosure-plugin'
import {
  NEEDS_A_HELPER, NEEDS_CONNECTION, NO_AGENT_STEP, NO_CUSTOM_PROMPT, NO_MODEL, NONE, NOT_ON_THIS_INSTALL,
  ON_THIS_INSTALL, OPTIONAL, UNNAMED, needsLine, personaPromptLine, plural, stepTitle,
} from './item-copy'

function PromptText({ prompt }: { prompt: CloudPrompt }) {
  if (!prompt.source || !prompt.text?.trim()) return <p className="text-sm text-muted-foreground">{NO_CUSTOM_PROMPT}</p>
  return (
    <div className="space-y-1.5">
      {prompt.source === 'persona' && <p className="text-xs text-muted-foreground">{personaPromptLine(prompt.personaName)}</p>}
      <TextBlock text={prompt.text} />
    </div>
  )
}

function hereNote(here: boolean): string | null {
  return here ? null : NOT_ON_THIS_INSTALL
}

function ModelLine({ model }: { model: AgentDisclosure['model'] }) {
  if (!model) return <p className="text-sm text-muted-foreground">{NO_MODEL}</p>
  return (
    <ul>
      <Line text={model.provider ? `${model.modelId} (${model.provider})` : model.modelId}
        note={model.here ? ON_THIS_INSTALL : NOT_ON_THIS_INSTALL} warn={!model.here} />
    </ul>
  )
}

function AgentSections({ disclosure }: { disclosure: AgentDisclosure }) {
  return (
    <>
      <Section title="Prompt"><PromptText prompt={disclosure.prompt} /></Section>
      <Section title="Model"><ModelLine model={disclosure.model} /></Section>
      <Section title="Apps">
        <Lines>{disclosure.apps.map((app, index) => (
          <Line key={index} text={app.name} note={app.needsConnection ? NEEDS_CONNECTION : null} warn />
        ))}</Lines>
      </Section>
      <Section title="Skills">
        <Lines>{disclosure.skills.map((skill, index) => (
          <Line key={index} text={skill.name} note={hereNote(skill.here)} warn />
        ))}</Lines>
      </Section>
      <Section title="Plugins">
        <Lines>{disclosure.plugins.map((plugin, index) => (
          <Line key={index} text={plugin.slug} note={hereNote(plugin.here)} warn />
        ))}</Lines>
      </Section>
    </>
  )
}

/**
 * "Step 2 · Ledger Helper", with "needs a helper" when the helper it names isn't here; a step that names
 * none (a document step) runs no agent.
 */
function StepHead({ step, index }: { step: PlaybookStep; index: number }) {
  return (
    <div className="flex flex-wrap items-center gap-2 text-sm">
      <span className="font-medium text-foreground">{stepTitle(step.order, index)}</span>
      <span className="min-w-0 break-words text-muted-foreground">{step.helper ?? NO_AGENT_STEP}</span>
      {step.needsHelper && <Note text={NEEDS_A_HELPER} warn />}
    </div>
  )
}

function PlaybookSections({ disclosure }: { disclosure: PlaybookDisclosure }) {
  return (
    <>
      <Section title="Steps">
        {disclosure.steps.length === 0 && <p className="text-sm text-muted-foreground">{NONE}</p>}
        <ol className="space-y-3">
          {disclosure.steps.map((step, index) => (
            <li key={index} className="space-y-1.5 rounded-xl border border-border/30 bg-secondary/30 p-3">
              <StepHead step={step} index={index} />
              {step.prompt && <TextBlock text={step.prompt} />}
            </li>
          ))}
        </ol>
      </Section>
      <Section title="Agents it adds">
        <Lines>{disclosure.agents.map((name, index) => <Line key={index} text={name} />)}</Lines>
      </Section>
      <Section title="Apps">
        <Lines>{disclosure.apps.map((app, index) => <Line key={index} text={app} />)}</Lines>
      </Section>
    </>
  )
}

function PackageSections({ disclosure }: { disclosure: PackageDisclosure }) {
  return (
    <>
      <Section title="Members">
        <Lines>{disclosure.members.map((member, index) => (
          <Line key={index} text={[member.name ?? UNNAMED, member.type && `(${member.type})`].filter(Boolean).join(' ')}
            note={hereNote(member.here)} warn />
        ))}</Lines>
      </Section>
      <Section title="Connections it asks for">
        <Lines>{disclosure.connects.map((connect, index) => (
          <Line key={index} text={[connect.app ?? UNNAMED, connect.note].filter(Boolean).join(': ')}
            note={connect.optional ? OPTIONAL : NEEDS_CONNECTION} warn={!connect.optional} />
        ))}</Lines>
      </Section>
      {disclosure.agents.map((agent, index) => (
        <Section key={index} title={`${agent.name}: prompt`}><PromptText prompt={agent.prompt} /></Section>
      ))}
      {disclosure.playbooks.length > 0 && (
        <Section title="Playbooks">
          <Lines>{disclosure.playbooks.map((playbook, index) => (
            <Line key={index} text={playbook.name} note={plural(playbook.steps, 'step', 'steps')} />
          ))}</Lines>
        </Section>
      )}
    </>
  )
}

function NeedsLines({ unresolved }: { unresolved: CloudUnresolved[] }) {
  if (unresolved.length === 0) return null
  return (
    <ul className="space-y-1" aria-label="What this install doesn't have">
      {unresolved.map((entry, index) => (
        <li key={index} className="text-sm text-warning">{needsLine(entry.kind, entry.name)}</li>
      ))}
    </ul>
  )
}

function KindSections({ disclosure }: { disclosure: CloudDisclosure }) {
  switch (disclosure.kind) {
    case 'agent': return <AgentSections disclosure={disclosure} />
    case 'playbook': return <PlaybookSections disclosure={disclosure} />
    case 'package': return <PackageSections disclosure={disclosure} />
    case 'plugin': return <PluginSections disclosure={disclosure} />
    default: return null
  }
}

export function DisclosureSections({ disclosure }: { disclosure: CloudDisclosure }) {
  return (
    <div className="space-y-5" data-testid="cloud-disclosure-sections">
      <KindSections disclosure={disclosure} />
      <NeedsLines unresolved={disclosure.unresolved} />
    </div>
  )
}
