'use client'

import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Badge } from '@/components/ui/badge'
import { Loader2, Plus, Trash2, Zap, CheckCircle2, XCircle, MessageSquare } from 'lucide-react'
import { toast } from 'sonner'
import { apiClient } from '@/lib/api-client'

interface ChannelConnection {
  id: string
  platform: string
  status: string
  message_count: number
  last_activity_at: string | null
  config: Record<string, string>
}

const PLATFORMS = [
  {
    id: 'whatsapp',
    name: 'WhatsApp',
    icon: 'WA',
    fields: [
      { key: 'phone_number_id', label: 'Phone Number ID', placeholder: 'From Meta Business' },
      { key: 'access_token', label: 'Access Token', placeholder: 'Permanent access token' },
      { key: 'verify_token', label: 'Verify Token (optional)', placeholder: 'Webhook verify token' }
    ],
    helpLink: 'https://developers.facebook.com/docs/whatsapp/cloud-api/get-started'
  },
  {
    id: 'telegram',
    name: 'Telegram',
    icon: 'TG',
    fields: [{ key: 'bot_token', label: 'Bot Token', placeholder: 'From @BotFather' }],
    helpLink: 'https://core.telegram.org/bots#botfather'
  },
  {
    id: 'slack',
    name: 'Slack',
    icon: 'SL',
    fields: [
      { key: 'bot_token', label: 'Bot Token', placeholder: 'xoxb-...' },
      { key: 'signing_secret', label: 'Signing Secret', placeholder: 'Slack app signing secret' },
      { key: 'app_token', label: 'App Token (Socket Mode)', placeholder: 'xapp-...' }
    ],
    helpLink: 'https://api.slack.com/apps'
  },
  {
    id: 'discord',
    name: 'Discord',
    icon: 'DC',
    fields: [{ key: 'bot_token', label: 'Bot Token', placeholder: 'Discord bot token' }],
    helpLink: 'https://discord.com/developers/applications'
  },
  {
    id: 'teams',
    name: 'MS Teams',
    icon: 'MT',
    fields: [
      { key: 'app_id', label: 'App ID', placeholder: 'Bot Framework App ID' },
      { key: 'app_password', label: 'App Password', placeholder: 'Bot Framework App Password' }
    ],
    helpLink: 'https://learn.microsoft.com/en-us/microsoftteams/platform/bots/how-to/create-a-bot-for-teams'
  },
  {
    id: 'google_chat',
    name: 'Google Chat',
    icon: 'GC',
    fields: [
      { key: 'service_account_json', label: 'Service Account JSON', placeholder: '{"type":"service_account",...}' }
    ],
    helpLink: 'https://developers.google.com/workspace/chat/overview'
  },
  {
    id: 'signal',
    name: 'Signal',
    icon: 'SG',
    fields: [
      { key: 'phone_number', label: 'Phone Number', placeholder: '+1234567890' },
      { key: 'api_url', label: 'signal-cli API URL', placeholder: 'http://localhost:8080' }
    ],
    helpLink: 'https://github.com/bbernhard/signal-cli-rest-api'
  },
  {
    id: 'imessage',
    name: 'iMessage',
    icon: 'iM',
    fields: [
      { key: 'server_url', label: 'BlueBubbles Server URL', placeholder: 'http://localhost:1234' },
      { key: 'password', label: 'Password', placeholder: 'BlueBubbles password' }
    ],
    helpLink: 'https://bluebubbles.app/'
  },
  {
    id: 'irc',
    name: 'IRC',
    icon: 'IR',
    fields: [
      { key: 'server', label: 'Server', placeholder: 'irc.libera.chat' },
      { key: 'port', label: 'Port', placeholder: '6697' },
      { key: 'nickname', label: 'Nickname', placeholder: 'automatos-bot' },
      { key: 'channels', label: 'Channels (comma-sep)', placeholder: '#automatos,#general' }
    ],
    helpLink: 'https://libera.chat/guides/connect'
  },
  {
    id: 'matrix',
    name: 'Matrix',
    icon: 'MX',
    fields: [
      { key: 'homeserver', label: 'Homeserver URL', placeholder: 'https://matrix.org' },
      { key: 'user_id', label: 'User ID', placeholder: '@bot:matrix.org' },
      { key: 'access_token', label: 'Access Token', placeholder: 'Matrix access token' }
    ],
    helpLink: 'https://matrix.org/docs/guides/usage-of-matrix-bots'
  },
  {
    id: 'line',
    name: 'LINE',
    icon: 'LN',
    fields: [
      { key: 'channel_access_token', label: 'Channel Access Token', placeholder: 'LINE channel access token' },
      { key: 'channel_secret', label: 'Channel Secret', placeholder: 'LINE channel secret' }
    ],
    helpLink: 'https://developers.line.biz/en/docs/messaging-api/'
  }
]

export function ChannelsSettingsTab() {
  const [channels, setChannels] = useState<ChannelConnection[]>([])
  const [loading, setLoading] = useState(true)
  const [connecting, setConnecting] = useState<string | null>(null)
  const [testing, setTesting] = useState<string | null>(null)
  const [newConfigs, setNewConfigs] = useState<Record<string, Record<string, string>>>({})
  const [fieldErrors, setFieldErrors] = useState<Record<string, Record<string, boolean>>>({})

  useEffect(() => {
    loadChannels()
  }, [])

  const loadChannels = async () => {
    setLoading(true)
    try {
      const data = await apiClient.request<ChannelConnection[] | { channels: ChannelConnection[] }>('/api/channels')
      const list = Array.isArray(data) ? data : (data as any).channels || []
      setChannels(list)
    } catch {
      setChannels([])
    } finally {
      setLoading(false)
    }
  }

  const connectChannel = async (platform: string) => {
    const config = newConfigs[platform] || {}
    const platformDef = PLATFORMS.find(p => p.id === platform)
    const requiredFields = (platformDef?.fields ?? []).filter(
      f => !f.label.toLowerCase().includes('(optional)')
    )
    const missing = requiredFields.filter(f => !(config[f.key] ?? '').trim())
    if (missing.length) {
      setFieldErrors(prev => ({
        ...prev,
        [platform]: missing.reduce<Record<string, boolean>>((acc, f) => {
          acc[f.key] = true
          return acc
        }, {}),
      }))
      toast.error(
        missing.length === 1
          ? `${missing[0].label} is required`
          : `Missing required fields: ${missing.map(f => f.label).join(', ')}`
      )
      return
    }
    setFieldErrors(prev => ({ ...prev, [platform]: {} }))
    setConnecting(platform)
    try {
      await apiClient.request('/api/channels', {
        method: 'POST',
        body: { platform, config } as any,
      })
      await loadChannels()
      setNewConfigs(prev => ({ ...prev, [platform]: {} }))
      toast.success(`${platformDef?.name ?? platform} connected`)
    } catch (err) {
      console.error('[Channels] connect error:', err)
      toast.error(err instanceof Error ? err.message : 'Failed to connect channel')
    } finally {
      setConnecting(null)
    }
  }

  const disconnectChannel = async (channelId: string) => {
    if (!confirm('Disconnect this channel?')) return
    try {
      await apiClient.request(`/api/channels/${channelId}`, { method: 'DELETE' })
      toast.success('Channel disconnected')
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to disconnect channel')
    }
    await loadChannels()
  }

  const testChannel = async (channelId: string) => {
    setTesting(channelId)
    try {
      const data = await apiClient.request<{
        status: string
        detail?: string
        bot_name?: string
        team?: { name?: string }
      }>(`/api/channels/${channelId}/test`, { method: 'POST' })
      // Backend returns ``{status: "connected", ...}`` on success and
      // ``{status: "error", detail: "..."}`` on failure. The previous
      // implementation checked for "ok" which never matched, so even
      // valid tokens reported "Test failed: Unknown error".
      if (data.status === 'connected') {
        const who = data.bot_name ?? data.team?.name
        toast.success(who ? `Connection successful — ${who}` : 'Connection successful')
      } else {
        toast.error(`Test failed: ${data.detail || 'Unknown error'}`)
      }
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Test failed')
    } finally {
      setTesting(null)
    }
  }

  const updateConfig = (platform: string, key: string, value: string) => {
    setNewConfigs(prev => ({
      ...prev,
      [platform]: { ...(prev[platform] || {}), [key]: value }
    }))
    // Clear the field error as soon as the user starts typing
    setFieldErrors(prev => {
      if (!prev[platform]?.[key]) return prev
      const next = { ...prev[platform] }
      delete next[key]
      return { ...prev, [platform]: next }
    })
  }

  const getConnectedChannel = (platform: string) => channels.find(c => c.platform === platform)

  if (loading) return <div className="flex items-center gap-2 p-8"><Loader2 className="h-4 w-4 animate-spin" /> Loading channels...</div>

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
        {PLATFORMS.map(platform => {
          const connected = getConnectedChannel(platform.id)
          return (
            <Card key={platform.id} className="glass-card card-glow hover:border-primary/20 transition-all duration-300 flex flex-col">
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-base flex items-center gap-2">
                    <span className="text-xl">{platform.icon}</span> {platform.name}
                  </CardTitle>
                  {connected ? (
                    <Badge variant={connected.status === 'active' ? 'default' : 'secondary'}>
                      {connected.status === 'active' ? <CheckCircle2 className="h-3 w-3 mr-1" /> : <XCircle className="h-3 w-3 mr-1" />}
                      {connected.status}
                    </Badge>
                  ) : (
                    <Badge variant="outline">Not connected</Badge>
                  )}
                </div>
              </CardHeader>
              <CardContent className="flex flex-col flex-1 pt-0">
                {connected ? (
                  <>
                    <div className="flex items-center gap-2 text-sm text-muted-foreground flex-1">
                      <MessageSquare className="h-3 w-3 shrink-0" />
                      <span>{connected.message_count} messages</span>
                      {connected.last_activity_at && (
                        <span>· {new Date(connected.last_activity_at).toLocaleDateString()}</span>
                      )}
                    </div>
                    <div className="flex gap-2 mt-4 pt-3 border-t border-border/30">
                      <Button size="sm" variant="outline" onClick={() => testChannel(connected.id)} disabled={testing === connected.id}>
                        {testing === connected.id ? <Loader2 className="h-3 w-3 animate-spin mr-1" /> : <Zap className="h-3 w-3 mr-1" />}
                        Test
                      </Button>
                      <Button size="sm" variant="outline" className="text-[hsl(var(--destructive))] hover:text-[hsl(var(--destructive))] hover:bg-[hsl(var(--destructive))]/10 hover:border-[hsl(var(--destructive))]/30" onClick={() => disconnectChannel(connected.id)}>
                        <Trash2 className="h-3 w-3 mr-1" /> Disconnect
                      </Button>
                    </div>
                  </>
                ) : (
                  <>
                    <div className="space-y-3 flex-1">
                      {platform.fields.map(field => {
                        const hasError = !!fieldErrors[platform.id]?.[field.key]
                        return (
                          <div key={field.key} className="space-y-1">
                            <Label className="text-xs">{field.label}</Label>
                            <Input
                              type="password"
                              placeholder={field.placeholder}
                              value={newConfigs[platform.id]?.[field.key] || ''}
                              onChange={e => updateConfig(platform.id, field.key, e.target.value)}
                              aria-invalid={hasError}
                              className={hasError ? 'border-[hsl(var(--destructive))] focus-visible:ring-[hsl(var(--destructive))]' : undefined}
                            />
                            {hasError && (
                              <p className="text-xs text-[hsl(var(--destructive))]">Required</p>
                            )}
                          </div>
                        )
                      })}
                    </div>
                    <div className="flex gap-2 mt-4 pt-3 border-t border-border/30">
                      <Button size="sm" variant="outline" onClick={() => connectChannel(platform.id)} disabled={connecting === platform.id}>
                        {connecting === platform.id ? <Loader2 className="h-3 w-3 animate-spin mr-1" /> : <Plus className="h-3 w-3 mr-1" />}
                        Connect
                      </Button>
                      <Button size="sm" variant="ghost" className="text-muted-foreground" asChild>
                        <a href={platform.helpLink} target="_blank" rel="noreferrer">Setup Guide</a>
                      </Button>
                    </div>
                  </>
                )}
              </CardContent>
            </Card>
          )
        })}
      </div>
    </div>
  )
}
