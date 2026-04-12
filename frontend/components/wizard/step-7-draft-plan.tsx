'use client'

import { useState } from 'react'
import { Sparkles, Users, ChevronDown, ChevronRight, MessageCircle } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import type { PlanResponse, PlanResponseAgent } from '@/hooks/use-wizard-api'

interface Step7Props {
  plan: PlanResponse
  onFinish: () => void
}

export function Step7DraftPlan({ plan, onFinish }: Step7Props) {
  const draft = plan.draft_plan

  return (
    <div className="space-y-4">
      <Card className="bg-secondary/30 border-border/30">
        <CardContent className="py-4">
          <div className="flex items-center gap-3">
            <Sparkles className="w-6 h-6 text-primary" />
            <div className="flex-1">
              <div className="font-medium">
                <span className="gradient-text">Mission Zero</span> Draft Plan
              </div>
              <div className="text-sm text-muted-foreground">
                {draft.proposed_agents.length} proposed agents · graph has{' '}
                {draft.graph_node_count} nodes
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Proposed agents */}
      <Card className="bg-secondary/30 border-border/30">
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <Users className="w-4 h-4 text-primary" />
            Proposed Team
          </CardTitle>
          <p className="text-sm text-muted-foreground">
            Each recommendation is grounded in evidence from your scraped content. Click a citation
            chip to see what drove it.
          </p>
        </CardHeader>
        <CardContent className="space-y-3">
          {draft.proposed_agents.map(agent => (
            <AgentCard key={agent.slug} agent={agent} />
          ))}
        </CardContent>
      </Card>

      {/* Open questions */}
      {draft.open_questions.length > 0 && (
        <Card className="bg-secondary/30 border-border/30">
          <CardHeader>
            <CardTitle className="text-base">Open Questions</CardTitle>
          </CardHeader>
          <CardContent className="space-y-1 text-sm">
            {draft.open_questions.map((q, i) => (
              <div key={i} className="text-muted-foreground">
                · {q}
              </div>
            ))}
          </CardContent>
        </Card>
      )}

      {/* CTA */}
      <Card className="bg-primary/10 border-primary/40">
        <CardContent className="py-4">
          <div className="flex items-center gap-3">
            <MessageCircle className="w-6 h-6 text-primary" />
            <div className="flex-1 text-sm">
              <div className="font-medium">Try it out</div>
              <div className="text-muted-foreground">
                Open chat and ask Auto: <em>&ldquo;tell me about my business&rdquo;</em> — your knowledge
                graph is now live.
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="flex justify-end">
        <Button onClick={onFinish}>Finish</Button>
      </div>
    </div>
  )
}

function AgentCard({ agent }: { agent: PlanResponseAgent }) {
  const [expanded, setExpanded] = useState(false)
  const [openCitation, setOpenCitation] = useState<string | null>(null)

  return (
    <div className="rounded-lg border border-border/30 bg-secondary/20">
      <button
        type="button"
        onClick={() => setExpanded(e => !e)}
        className="w-full text-left p-4 flex items-start gap-3"
      >
        {expanded ? (
          <ChevronDown className="w-4 h-4 mt-1 flex-shrink-0" />
        ) : (
          <ChevronRight className="w-4 h-4 mt-1 flex-shrink-0" />
        )}
        <div className="flex-1">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="font-medium">{agent.name}</span>
            <Badge variant="outline" className="text-xs">
              {agent.team}
            </Badge>
            {agent.citations.length > 0 && (
              <Badge variant="outline" className="text-xs">
                {agent.citations.length} citation{agent.citations.length === 1 ? '' : 's'}
              </Badge>
            )}
          </div>
          <div className="text-xs text-muted-foreground mt-0.5">{agent.job_title}</div>
          <div className="text-sm text-muted-foreground mt-1">{agent.rationale}</div>
        </div>
      </button>

      {expanded && (
        <div className="px-4 pb-4 space-y-3">
          <div className="text-sm">
            <div className="text-xs uppercase tracking-wide text-muted-foreground mb-1">Persona</div>
            <div className="text-muted-foreground">{agent.persona}</div>
          </div>

          <div className="flex gap-4 text-xs">
            <div>
              <div className="uppercase tracking-wide text-muted-foreground mb-1">Skills</div>
              <div className="flex flex-wrap gap-1">
                {agent.skills.map(s => (
                  <Badge key={s} variant="outline" className="text-xs">
                    {s}
                  </Badge>
                ))}
              </div>
            </div>
            <div>
              <div className="uppercase tracking-wide text-muted-foreground mb-1">Tools</div>
              <div className="flex flex-wrap gap-1">
                {agent.tools.map(t => (
                  <Badge key={t} variant="outline" className="text-xs">
                    {t}
                  </Badge>
                ))}
              </div>
            </div>
          </div>

          {agent.citations.length > 0 && (
            <div>
              <div className="text-xs uppercase tracking-wide text-muted-foreground mb-2">
                Evidence from your site
              </div>
              <div className="flex flex-wrap gap-1">
                {agent.citations.map(c => (
                  <button
                    key={c.id}
                    type="button"
                    onClick={() => setOpenCitation(openCitation === c.id ? null : c.id)}
                    className={`px-2 py-1 text-xs rounded border transition-all ${
                      openCitation === c.id
                        ? 'border-primary bg-primary/10'
                        : 'border-border/40 bg-secondary/30 hover:border-border/60'
                    }`}
                  >
                    {c.label}
                  </button>
                ))}
              </div>
              {openCitation && (
                <div className="mt-2 p-3 rounded-md bg-secondary/40 border border-border/30 text-xs text-muted-foreground">
                  {agent.citations.find(c => c.id === openCitation)?.snippet ||
                    'No snippet available — node label only.'}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
