'use client';

/**
 * /styleguide — Living spec for the Studio rebrand.
 *
 * Documents tokens, primitives, states, and Studio utility classes so
 * engineers and CD can see the system in one place. Always renders in
 * .studio scope regardless of the user's active theme preference.
 *
 * PRD §3 deliverable #6. Updated as primitives land during Phase 1+.
 */

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import { Switch } from '@/components/ui/switch';
import { Checkbox } from '@/components/ui/checkbox';
import { Label } from '@/components/ui/label';
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from '@/components/ui/select';
import { GlossaryTooltip } from '@/components/ui/glossary-tooltip';
import { PageHeader } from '@/components/shared/page-header';
import { StatusBadge } from '@/components/shared/status-badge';
import { Download, Plus, Check, Search, AlertTriangle, Info } from 'lucide-react';

// Force Studio scope on this route regardless of active theme. Lets us see
// the spec even when the user is browsing in classic.
const FORCE_STUDIO = 'studio';

function Section({
  eyebrow,
  title,
  children,
}: {
  eyebrow: string;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section className="space-y-4 border-t border-border pt-8">
      <div>
        <p className="studio-eyebrow text-[11px] uppercase tracking-[0.08em] text-muted-foreground font-mono font-semibold mb-1">
          {eyebrow}
        </p>
        <h2 className="text-2xl font-serif font-medium">{title}</h2>
      </div>
      {children}
    </section>
  );
}

function Swatch({
  name,
  cssVar,
  hex,
  note,
}: {
  name: string;
  cssVar: string;
  hex: string;
  note?: string;
}) {
  return (
    <div className="space-y-2">
      <div
        className="h-20 rounded-md border border-border"
        style={{ background: `hsl(var(${cssVar}))` }}
      />
      <div className="space-y-0.5">
        <div className="text-xs font-medium">{name}</div>
        <div className="text-[10px] font-mono text-muted-foreground">{cssVar}</div>
        <div className="text-[10px] font-mono text-muted-foreground">{hex}</div>
        {note && <div className="text-[10px] text-muted-foreground italic">{note}</div>}
      </div>
    </div>
  );
}

export default function StyleguidePage() {
  return (
    <div className={FORCE_STUDIO}>
      <div className="min-h-screen bg-background text-foreground">
        <div className="max-w-6xl mx-auto px-8 py-12 space-y-12">
          {/* Page header demonstrates editorial-first pattern */}
          <PageHeader
            eyebrow="Phase 1 · living spec"
            title="Studio"
            titleAccent="Styleguide"
            lede={
              <>
                Tokens, primitives, and the semantic vocabulary that runs Ledger Studio across the
                platform. Always renders in <code className="font-mono text-xs">.studio</code> scope
                — flip the theme toggle in your menu to compare with the classic system.
              </>
            }
            actions={
              <>
                <Button variant="outline" size="sm">
                  <Download className="h-4 w-4 mr-1" />
                  Export tokens
                </Button>
                <Button size="sm">
                  <Plus className="h-4 w-4 mr-1" />
                  New page
                </Button>
              </>
            }
          />

          {/* Semantic lock — the rule that runs everything */}
          <Section eyebrow="Section 1 of 8" title="Semantic colour lock">
            <p className="studio-lede max-w-[70ch] text-sm text-muted-foreground">
              Five colours, each with one semantic job. Never paint a destructive action olive.
              Never paint a positive primary orange. The lock is the contract.
            </p>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
              <Swatch
                name="Near-black"
                cssVar="--primary"
                hex="#1a1814"
                note="Positive primary CTA"
              />
              <Swatch
                name="Burnt orange"
                cssVar="--accent"
                hex="#c44a1a"
                note="Consequence / destructive"
              />
              <Swatch name="Olive" cssVar="--success" hex="#5b6f3a" note="Good / success" />
              <Swatch name="Navy" cssVar="--info" hex="#1d3658" note="Info / queued" />
              <Swatch name="Tan" cssVar="--border" hex="#dcd2bd" note="Neutral / border" />
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <Swatch name="Cream paper" cssVar="--background" hex="#f4eee2" />
              <Swatch name="White card" cssVar="--card" hex="#ffffff" />
              <Swatch name="Warm ink" cssVar="--foreground" hex="#1a1814" />
              <Swatch
                name="Muted ink"
                cssVar="--muted-foreground"
                hex="#5a5448"
                note="Body copy, ledes"
              />
            </div>
          </Section>

          {/* Typography */}
          <Section eyebrow="Section 2 of 8" title="Typography">
            <div className="space-y-3">
              <div>
                <p className="text-xs font-mono text-muted-foreground mb-1">
                  Display · serif · text-5xl · tracking tight
                </p>
                <h1 className="text-5xl font-serif font-medium tracking-tight">
                  Your workforce is running.
                </h1>
              </div>
              <div>
                <p className="text-xs font-mono text-muted-foreground mb-1">
                  H1 · serif · text-3xl
                </p>
                <h1 className="text-3xl font-serif font-medium">Launch · Q3 product update</h1>
              </div>
              <div>
                <p className="text-xs font-mono text-muted-foreground mb-1">
                  H2 · serif · text-2xl
                </p>
                <h2 className="text-2xl font-serif font-medium">Activity log</h2>
              </div>
              <div>
                <p className="text-xs font-mono text-muted-foreground mb-1">
                  Lede · sans · text-sm · 70ch · muted
                </p>
                <p className="studio-lede max-w-[70ch] text-sm text-muted-foreground leading-relaxed">
                  Every tool call, handoff, router decision, and memory write. Tailed live,
                  replayable to the second, and posted straight to the books.
                </p>
              </div>
              <div>
                <p className="text-xs font-mono text-muted-foreground mb-1">
                  Mono · JetBrains Mono · 11px · 0.04em tracking
                </p>
                <p className="font-mono text-[11px] tracking-wider">
                  09:46:48.103 · sentinel · tool.call · github.create_pr · HTTP 422
                </p>
              </div>
              <div>
                <p className="text-xs font-mono text-muted-foreground mb-1">Eyebrow · mono uppercase</p>
                <p className="studio-eyebrow text-[11px] uppercase tracking-[0.08em] text-muted-foreground font-mono font-semibold">
                  Execution log · today, 09:00 → now · tick 5s
                </p>
              </div>
            </div>
          </Section>

          {/* Buttons — hierarchy and states */}
          <Section eyebrow="Section 3 of 8" title="Buttons">
            <div className="space-y-4">
              <div className="space-y-2">
                <p className="studio-eyebrow text-[11px] uppercase tracking-[0.08em] text-muted-foreground font-mono font-semibold">
                  Hierarchy
                </p>
                <div className="flex flex-wrap gap-2 items-center">
                  <Button>Positive primary</Button>
                  <Button variant="secondary">Secondary</Button>
                  <Button variant="outline">Outline</Button>
                  <Button variant="ghost">Ghost</Button>
                  <Button variant="destructive">Consequence · bypass &amp; merge</Button>
                  <Button variant="link">Link</Button>
                  <Button disabled>Disabled</Button>
                </div>
              </div>
              <div className="space-y-2">
                <p className="studio-eyebrow text-[11px] uppercase tracking-[0.08em] text-muted-foreground font-mono font-semibold">
                  Sizes + icon
                </p>
                <div className="flex flex-wrap gap-2 items-center">
                  <Button size="sm">Small</Button>
                  <Button>Default</Button>
                  <Button size="lg">Large</Button>
                  <Button>
                    <Plus className="h-4 w-4 mr-1" />
                    With icon
                  </Button>
                  <Button size="icon" aria-label="Search">
                    <Search className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            </div>
          </Section>

          {/* Inputs + form */}
          <Section eyebrow="Section 4 of 8" title="Inputs + form">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-w-2xl">
              <div className="space-y-1.5">
                <Label htmlFor="sg-input">Default input</Label>
                <Input id="sg-input" placeholder="msn_8f3a" />
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="sg-input-2">Disabled</Label>
                <Input id="sg-input-2" placeholder="—" disabled />
              </div>
              <div className="space-y-1.5 md:col-span-2">
                <Label htmlFor="sg-textarea">Textarea</Label>
                <Textarea
                  id="sg-textarea"
                  placeholder="Pull together the Q3 product update — last week's logs into a one-pager…"
                  rows={3}
                />
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="sg-select">Select</Label>
                <Select>
                  <SelectTrigger id="sg-select">
                    <SelectValue placeholder="Claude 3.5 Sonnet" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="sonnet">Claude 3.5 Sonnet</SelectItem>
                    <SelectItem value="haiku">Claude 3.5 Haiku</SelectItem>
                    <SelectItem value="gpt4">GPT-4</SelectItem>
                    <SelectItem value="deepseek">DeepSeek V3</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-3 self-end">
                <div className="flex items-center gap-2">
                  <Switch id="sg-switch" defaultChecked />
                  <Label htmlFor="sg-switch">Active · accepting work</Label>
                </div>
                <div className="flex items-center gap-2">
                  <Checkbox id="sg-check" />
                  <Label htmlFor="sg-check">Auto-retry on failure</Label>
                </div>
              </div>
            </div>
          </Section>

          {/* Badges + pills */}
          <Section eyebrow="Section 5 of 8" title="Badges + status pills">
            <div className="space-y-4">
              <div className="space-y-2">
                <p className="studio-eyebrow text-[11px] uppercase tracking-[0.08em] text-muted-foreground font-mono font-semibold">
                  Studio pills · 6 semantic tones · square 2px
                </p>
                <div className="flex flex-wrap gap-2 items-center">
                  <span className="studio-pill ok">● OK</span>
                  <span className="studio-pill warn">● WARN</span>
                  <span className="studio-pill err">● ERR</span>
                  <span className="studio-pill info">● QUEUED</span>
                  <span className="studio-pill brand">msn_8f3a</span>
                  <span className="studio-pill muted">DRAFT</span>
                </div>
              </div>
              <div className="space-y-2">
                <p className="studio-eyebrow text-[11px] uppercase tracking-[0.08em] text-muted-foreground font-mono font-semibold">
                  Shadcn Badge — token-driven
                </p>
                <div className="flex flex-wrap gap-2 items-center">
                  <Badge>Default</Badge>
                  <Badge variant="secondary">Secondary</Badge>
                  <Badge variant="outline">Outline</Badge>
                  <Badge variant="destructive">Destructive</Badge>
                </div>
              </div>
              <div className="space-y-2">
                <p className="studio-eyebrow text-[11px] uppercase tracking-[0.08em] text-muted-foreground font-mono font-semibold">
                  StatusBadge — token-driven
                </p>
                <div className="flex flex-wrap gap-2 items-center">
                  <StatusBadge status="success" dot>
                    Verified
                  </StatusBadge>
                  <StatusBadge status="warning" dot>
                    Pending
                  </StatusBadge>
                  <StatusBadge status="error" dot>
                    Failed
                  </StatusBadge>
                  <StatusBadge status="info" dot>
                    Routed
                  </StatusBadge>
                  <StatusBadge status="neutral">Neutral</StatusBadge>
                </div>
              </div>
            </div>
          </Section>

          {/* Status pip vocabulary */}
          <Section eyebrow="Section 6 of 8" title="Status icon vocabulary · locked">
            <p className="studio-lede max-w-[70ch] text-sm text-muted-foreground">
              Six glyphs, six meanings. Used inside <code className="font-mono">.studio-pip</code>{' '}
              circles in mission DAGs, audit row prefixes, and the activity ticker.
            </p>
            <div className="grid grid-cols-6 gap-4">
              {[
                { ch: '✓', label: 'done', cls: 'done' },
                { ch: '!', label: 'error', cls: 'error' },
                { ch: '↻', label: 'queued', cls: 'queued' },
                { ch: '◐', label: 'running', cls: 'running' },
                { ch: '·', label: 'pending', cls: 'pending' },
                { ch: '◦', label: 'paused', cls: 'paused' },
              ].map((p) => (
                <div key={p.label} className="text-center space-y-1.5">
                  <div className={`studio-pip ${p.cls} mx-auto`} style={{ width: 36, height: 36, fontSize: 16 }}>
                    {p.ch}
                  </div>
                  <div className="text-[10.5px] font-mono uppercase tracking-wider text-muted-foreground">
                    {p.label}
                  </div>
                </div>
              ))}
            </div>
          </Section>

          {/* Studio panel + cards */}
          <Section eyebrow="Section 7 of 8" title="Panels + cards">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="studio-panel">
                <div className="studio-panel-head">
                  Live taps · last 5m · <span className="studio-pill ok">MSN_8F3A</span>
                </div>
                <div className="p-4 text-sm space-y-2">
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-xs text-muted-foreground">09:46:30</span>
                    <span className="studio-pill ok">OK</span>
                    <span className="font-serif font-semibold">Sentinel</span>
                    <span className="font-mono text-xs text-muted-foreground">review.pass</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-xs text-muted-foreground">09:46:48</span>
                    <span className="studio-pill err">ERR</span>
                    <span className="font-serif font-semibold">Sentinel</span>
                    <span className="font-mono text-xs text-muted-foreground">github.create_pr</span>
                  </div>
                </div>
              </div>
              <Card>
                <CardHeader>
                  <CardTitle>Shadcn Card</CardTitle>
                  <CardDescription>Token-driven; inherits Studio paper-on-cream.</CardDescription>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground">
                    Cards in Studio are flat paper with 1px tan border. Hover lifts to a soft
                    olive-tinted shadow per the microspec.
                  </p>
                </CardContent>
              </Card>
            </div>
          </Section>

          {/* Tabs + glossary tooltip */}
          <Section eyebrow="Section 8 of 8" title="Tabs + glossary tooltips">
            <Tabs defaultValue="overview">
              <TabsList>
                <TabsTrigger value="overview">Overview</TabsTrigger>
                <TabsTrigger value="activity">Activity</TabsTrigger>
                <TabsTrigger value="spend">Spend</TabsTrigger>
              </TabsList>
              <TabsContent value="overview" className="pt-4">
                <p className="studio-lede max-w-[70ch] text-sm text-muted-foreground">
                  A <GlossaryTooltip term="mission">mission</GlossaryTooltip> is a single piece of
                  work the platform runs end-to-end. Each one has an{' '}
                  <GlossaryTooltip term="agent">agent</GlossaryTooltip> picked by the{' '}
                  <GlossaryTooltip term="router">router</GlossaryTooltip> (defaulting to{' '}
                  <GlossaryTooltip term="t25">T2.5</GlossaryTooltip>) plus a record of every{' '}
                  <GlossaryTooltip term="handoff">handoff</GlossaryTooltip>. The output is a{' '}
                  <GlossaryTooltip term="deliverable">deliverable</GlossaryTooltip>.
                </p>
                <p className="text-xs text-muted-foreground mt-3 italic">
                  Hover any underlined term. After 3 sightings per browser the tooltip suppresses
                  itself.
                </p>
              </TabsContent>
              <TabsContent value="activity">
                <p className="text-sm text-muted-foreground">Activity tab content.</p>
              </TabsContent>
              <TabsContent value="spend">
                <p className="text-sm text-muted-foreground">Spend tab content.</p>
              </TabsContent>
            </Tabs>
          </Section>

          {/* Footer */}
          <div className="border-t border-border pt-6 text-xs text-muted-foreground font-mono space-y-1">
            <p>
              Living spec · Phase 1 · {new Date().toISOString().slice(0, 10)} · auto-renders in{' '}
              <span className="font-semibold">.studio</span> scope
            </p>
            <p>
              PRD: <span className="font-semibold">PRD_PLATFORM_REDESIGN.md</span> · Microspec:{' '}
              <span className="font-semibold">DUMPING AREA/phase1/microspec.jsx</span> · Round 3:{' '}
              <span className="font-semibold">DUMPING AREA/DesignKIT/round3/</span>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
