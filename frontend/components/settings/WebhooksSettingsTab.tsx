'use client'

import { useState, useEffect } from 'react'
import { Copy, Check, Webhook, ArrowRight, Save, Loader2, Eye, EyeOff, MessageCircle, Hash, Phone } from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import { useWorkspace } from '@/components/workspace-provider'
import { useAuth } from '@clerk/nextjs'

interface IntegrationStatus {
  configured: boolean
  masked?: string
}

export default function WebhooksSettingsTab() {
  const { workspace, isLoading } = useWorkspace()
  const { getToken } = useAuth()
  const [copiedField, setCopiedField] = useState<string | null>(null)

  // Integration form state
  const [telegramToken, setTelegramToken] = useState('')
  const [slackToken, setSlackToken] = useState('')
  const [whatsappPhoneId, setWhatsappPhoneId] = useState('')
  const [whatsappToken, setWhatsappToken] = useState('')

  // UI state
  const [saving, setSaving] = useState(false)
  const [saveSuccess, setSaveSuccess] = useState(false)
  const [showTokens, setShowTokens] = useState<Record<string, boolean>>({})
  const [integrationStatus, setIntegrationStatus] = useState<Record<string, IntegrationStatus>>({})

  const apiUrl = process.env.NEXT_PUBLIC_API_URL

  // Load current integration status
  useEffect(() => {
    if (!workspace) return
    loadIntegrationStatus()
  }, [workspace?.id])

  const loadIntegrationStatus = async () => {
    try {
      const token = await getToken()
      const res = await fetch(`${apiUrl}/api/workspaces/current/integrations`, {
        headers: { Authorization: `Bearer ${token}` },
      })
      if (res.ok) {
        const data = await res.json()
        setIntegrationStatus(data)
      }
    } catch {
      // Ignore — integrations may not be configured yet
    }
  }

  const handleCopy = (text: string, field: string) => {
    navigator.clipboard.writeText(text)
    setCopiedField(field)
    setTimeout(() => setCopiedField(null), 2000)
  }

  const handleSaveIntegrations = async () => {
    setSaving(true)
    setSaveSuccess(false)
    try {
      const token = await getToken()
      const payload: Record<string, string> = {}

      // Only send non-empty values (don't overwrite with blanks)
      if (telegramToken.trim()) payload.telegram_bot_token = telegramToken.trim()
      if (slackToken.trim()) payload.slack_bot_token = slackToken.trim()
      if (whatsappPhoneId.trim()) payload.whatsapp_phone_number_id = whatsappPhoneId.trim()
      if (whatsappToken.trim()) payload.whatsapp_access_token = whatsappToken.trim()

      const res = await fetch(`${apiUrl}/api/workspaces/current/integrations`, {
        method: 'PUT',
        headers: {
          Authorization: `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      })

      if (res.ok) {
        setSaveSuccess(true)
        // Clear form fields (they'll show as "configured" via status)
        setTelegramToken('')
        setSlackToken('')
        setWhatsappPhoneId('')
        setWhatsappToken('')
        // Reload status
        await loadIntegrationStatus()
        setTimeout(() => setSaveSuccess(false), 3000)
      }
    } catch (err) {
      console.error('Failed to save integrations:', err)
    } finally {
      setSaving(false)
    }
  }

  const toggleShow = (key: string) => {
    setShowTokens(prev => ({ ...prev, [key]: !prev[key] }))
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
            </>
          ) : (
            <p className="text-sm text-muted-foreground">
              Webhook URL is not yet configured. Try refreshing the page.
            </p>
          )}
        </CardContent>
      </Card>

      {/* Platform Integrations */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Platform Integrations</CardTitle>
          <CardDescription>
            Configure bot tokens so the orchestrator can reply back into messaging platforms.
            When a message arrives via the workspace webhook, the response will be sent back to the originating chat.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Telegram */}
          <div className="space-y-3 p-4 rounded-lg border border-border/50">
            <div className="flex items-center gap-2">
              <MessageCircle className="w-4 h-4 text-blue-400" />
              <Label className="font-semibold">Telegram</Label>
              {integrationStatus.telegram_bot_token?.configured && (
                <span className="text-xs text-green-500 bg-green-500/10 px-2 py-0.5 rounded-full">Connected</span>
              )}
            </div>
            <p className="text-xs text-muted-foreground">
              Create a bot via <a href="https://t.me/BotFather" target="_blank" rel="noopener noreferrer" className="text-primary hover:underline">@BotFather</a>,
              then set the webhook URL to your workspace webhook above.
            </p>
            <div className="flex items-center gap-2">
              <Input
                type={showTokens.telegram ? 'text' : 'password'}
                value={telegramToken}
                onChange={(e) => setTelegramToken(e.target.value)}
                placeholder={integrationStatus.telegram_bot_token?.configured ? `Current: ${integrationStatus.telegram_bot_token.masked}` : 'Bot token from @BotFather'}
                className="font-mono text-xs"
              />
              <Button variant="ghost" size="icon" onClick={() => toggleShow('telegram')} className="shrink-0">
                {showTokens.telegram ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </Button>
            </div>
            {integrationStatus.telegram_bot_token?.configured && webhookUrl && (
              <div className="text-xs text-muted-foreground bg-muted/50 rounded p-2 font-mono">
                <span className="text-primary">Setup:</span> curl -X POST &quot;https://api.telegram.org/bot{'<token>'}/setWebhook?url={webhookUrl}&quot;
              </div>
            )}
          </div>

          {/* Slack */}
          <div className="space-y-3 p-4 rounded-lg border border-border/50">
            <div className="flex items-center gap-2">
              <Hash className="w-4 h-4 text-purple-400" />
              <Label className="font-semibold">Slack</Label>
              {integrationStatus.slack_bot_token?.configured && (
                <span className="text-xs text-green-500 bg-green-500/10 px-2 py-0.5 rounded-full">Connected</span>
              )}
            </div>
            <p className="text-xs text-muted-foreground">
              Create a Slack app, add the <code className="text-xs">chat:write</code> scope, install to workspace,
              and configure Event Subscriptions to point to your workspace webhook above.
            </p>
            <div className="flex items-center gap-2">
              <Input
                type={showTokens.slack ? 'text' : 'password'}
                value={slackToken}
                onChange={(e) => setSlackToken(e.target.value)}
                placeholder={integrationStatus.slack_bot_token?.configured ? `Current: ${integrationStatus.slack_bot_token.masked}` : 'Bot User OAuth Token (xoxb-...)'}
                className="font-mono text-xs"
              />
              <Button variant="ghost" size="icon" onClick={() => toggleShow('slack')} className="shrink-0">
                {showTokens.slack ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </Button>
            </div>
          </div>

          {/* WhatsApp */}
          <div className="space-y-3 p-4 rounded-lg border border-border/50">
            <div className="flex items-center gap-2">
              <Phone className="w-4 h-4 text-green-400" />
              <Label className="font-semibold">WhatsApp Business</Label>
              {integrationStatus.whatsapp_access_token?.configured && (
                <span className="text-xs text-green-500 bg-green-500/10 px-2 py-0.5 rounded-full">Connected</span>
              )}
            </div>
            <p className="text-xs text-muted-foreground">
              From the Meta Developer Console, get your Phone Number ID and permanent access token.
              Set the webhook URL in your WhatsApp Business app settings.
            </p>
            <div className="grid grid-cols-2 gap-2">
              <div>
                <Label className="text-xs text-muted-foreground">Phone Number ID</Label>
                <Input
                  value={whatsappPhoneId}
                  onChange={(e) => setWhatsappPhoneId(e.target.value)}
                  placeholder={integrationStatus.whatsapp_phone_number_id?.configured ? `Current: ${integrationStatus.whatsapp_phone_number_id.masked}` : 'Phone Number ID'}
                  className="font-mono text-xs"
                />
              </div>
              <div>
                <Label className="text-xs text-muted-foreground">Access Token</Label>
                <div className="flex items-center gap-2">
                  <Input
                    type={showTokens.whatsapp ? 'text' : 'password'}
                    value={whatsappToken}
                    onChange={(e) => setWhatsappToken(e.target.value)}
                    placeholder={integrationStatus.whatsapp_access_token?.configured ? `Current: ${integrationStatus.whatsapp_access_token.masked}` : 'Permanent access token'}
                    className="font-mono text-xs"
                  />
                  <Button variant="ghost" size="icon" onClick={() => toggleShow('whatsapp')} className="shrink-0">
                    {showTokens.whatsapp ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </Button>
                </div>
              </div>
            </div>
          </div>

          {/* Save Button */}
          <div className="flex items-center gap-3">
            <Button
              onClick={handleSaveIntegrations}
              disabled={saving || (!telegramToken && !slackToken && !whatsappPhoneId && !whatsappToken)}
            >
              {saving ? (
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              ) : (
                <Save className="w-4 h-4 mr-2" />
              )}
              {saving ? 'Saving...' : 'Save Integrations'}
            </Button>
            {saveSuccess && (
              <span className="text-sm text-green-500 flex items-center gap-1">
                <Check className="w-4 h-4" /> Saved
              </span>
            )}
          </div>
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
            href="/workflows"
            className="inline-flex items-center gap-1.5 text-sm text-primary hover:underline"
          >
            Go to Recipes
            <ArrowRight className="w-3.5 h-3.5" />
          </a>
        </CardContent>
      </Card>
    </div>
  )
}
