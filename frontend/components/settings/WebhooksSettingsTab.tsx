'use client'

import { useState } from 'react'
import { Copy, Check, Webhook, ArrowRight } from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { useWorkspace } from '@/components/workspace-provider'

export default function WebhooksSettingsTab() {
  const { workspace, isLoading } = useWorkspace()
  const [copiedField, setCopiedField] = useState<string | null>(null)

  const handleCopy = (text: string, field: string) => {
    navigator.clipboard.writeText(text)
    setCopiedField(field)
    setTimeout(() => setCopiedField(null), 2000)
  }

  if (isLoading) {
    return <div className="text-sm text-muted-foreground p-4">Loading workspace...</div>
  }

  const webhookUrl = workspace?.webhookUrl

  return (
    <div className="space-y-6">
      {/* Workspace Webhook */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Webhook className="w-5 h-5" />
            Workspace Webhook
          </CardTitle>
          <CardDescription>
            Send any message to this URL. The orchestrator will route it to the right agent based on your routing rules.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {webhookUrl ? (
            <>
              <div className="flex items-center gap-2">
                <Input readOnly value={webhookUrl} className="font-mono text-xs" />
                <Button
                  variant="outline"
                  size="icon"
                  onClick={() => handleCopy(webhookUrl, 'webhook')}
                  className={`shrink-0 transition-colors ${copiedField === 'webhook' ? 'text-green-500 border-green-500' : ''}`}
                >
                  {copiedField === 'webhook' ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
                </Button>
              </div>

              <div className="rounded-lg bg-muted p-4 font-mono text-xs space-y-1">
                <div><span className="text-primary font-semibold">POST</span> {webhookUrl}</div>
                <div className="text-muted-foreground">Content-Type: application/json</div>
                <div className="mt-2 text-muted-foreground">{'{'}</div>
                <div className="text-muted-foreground pl-4">{'"message": "Check my latest emails"'}</div>
                <div className="text-muted-foreground">{'}'}</div>
              </div>

              <div className="text-xs text-muted-foreground space-y-1">
                <p>The webhook accepts any JSON body. Common fields:</p>
                <ul className="list-disc list-inside pl-2 space-y-0.5">
                  <li><code className="text-xs">message</code> / <code className="text-xs">text</code> / <code className="text-xs">content</code> — The message to route</li>
                  <li><code className="text-xs">agent_id</code> — Force routing to a specific agent</li>
                  <li><code className="text-xs">source</code> / <code className="text-xs">channel</code> — Metadata for routing rules</li>
                </ul>
              </div>

              <div className="rounded-lg border border-blue-500/20 bg-blue-500/5 p-3 text-xs text-muted-foreground">
                To connect messaging platforms (WhatsApp, Telegram, Slack, Discord, etc.), go to the <strong>Channels</strong> tab.
              </div>
            </>
          ) : (
            <p className="text-sm text-muted-foreground">
              Webhook URL is not yet configured. Try refreshing the page.
            </p>
          )}
        </CardContent>
      </Card>

      {/* Recipe Webhooks Reference */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Recipe Webhooks</CardTitle>
          <CardDescription>
            Each recipe with a &quot;Triggered&quot; execution type gets its own unique webhook URL.
            Configure triggers on individual recipes to get task-specific webhooks.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <a
            href="/assignments?tab=playbooks"
            className="inline-flex items-center gap-1.5 text-sm text-primary hover:underline"
          >
            Go to Playbooks
            <ArrowRight className="w-3.5 h-3.5" />
          </a>
        </CardContent>
      </Card>
    </div>
  )
}
