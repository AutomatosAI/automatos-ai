'use client'

import { useState, useEffect, useCallback } from 'react'
import {
  Volume2,
  Plus,
  Trash2,
  Play,
  Square,
  Upload,
  Loader2,
  Star,
  Mic,
  Radio,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Badge } from '@/components/ui/badge'
import { Switch } from '@/components/ui/switch'
import { Progress } from '@/components/ui/progress'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog'
import { apiClient } from '@/lib/api-client'
import { toast } from 'sonner'

// Known voices per provider — full Kokoro-82M catalogue
const KOKORO_VOICES = [
  // American English — Female
  { id: 'af_heart', label: 'Heart (American F)' },
  { id: 'af_alloy', label: 'Alloy (American F)' },
  { id: 'af_aoede', label: 'Aoede (American F)' },
  { id: 'af_bella', label: 'Bella (American F)' },
  { id: 'af_jessica', label: 'Jessica (American F)' },
  { id: 'af_kore', label: 'Kore (American F)' },
  { id: 'af_nicole', label: 'Nicole (American F)' },
  { id: 'af_nova', label: 'Nova (American F)' },
  { id: 'af_river', label: 'River (American F)' },
  { id: 'af_sarah', label: 'Sarah (American F)' },
  { id: 'af_sky', label: 'Sky (American F)' },
  // American English — Male
  { id: 'am_adam', label: 'Adam (American M)' },
  { id: 'am_echo', label: 'Echo (American M)' },
  { id: 'am_eric', label: 'Eric (American M)' },
  { id: 'am_fenrir', label: 'Fenrir (American M)' },
  { id: 'am_liam', label: 'Liam (American M)' },
  { id: 'am_michael', label: 'Michael (American M)' },
  { id: 'am_onyx', label: 'Onyx (American M)' },
  { id: 'am_puck', label: 'Puck (American M)' },
  { id: 'am_santa', label: 'Santa (American M)' },
  // British English — Female
  { id: 'bf_alice', label: 'Alice (British F)' },
  { id: 'bf_emma', label: 'Emma (British F)' },
  { id: 'bf_isabella', label: 'Isabella (British F)' },
  { id: 'bf_lily', label: 'Lily (British F)' },
  // British English — Male
  { id: 'bm_daniel', label: 'Daniel (British M)' },
  { id: 'bm_fable', label: 'Fable (British M)' },
  { id: 'bm_george', label: 'George (British M)' },
  { id: 'bm_lewis', label: 'Lewis (British M)' },
  // Spanish
  { id: 'ef_dora', label: 'Dora (Spanish F)' },
  { id: 'em_alex', label: 'Alex (Spanish M)' },
  { id: 'em_santa', label: 'Santa (Spanish M)' },
  // French
  { id: 'ff_siwis', label: 'Siwis (French F)' },
  // Hindi
  { id: 'hf_alpha', label: 'Alpha (Hindi F)' },
  { id: 'hf_beta', label: 'Beta (Hindi F)' },
  { id: 'hm_omega', label: 'Omega (Hindi M)' },
  { id: 'hm_psi', label: 'Psi (Hindi M)' },
  // Italian
  { id: 'if_sara', label: 'Sara (Italian F)' },
  { id: 'im_nicola', label: 'Nicola (Italian M)' },
  // Japanese
  { id: 'jf_alpha', label: 'Alpha (Japanese F)' },
  { id: 'jf_gongitsune', label: 'Gongitsune (Japanese F)' },
  { id: 'jf_nezumi', label: 'Nezumi (Japanese F)' },
  { id: 'jf_tebukuro', label: 'Tebukuro (Japanese F)' },
  { id: 'jm_kumo', label: 'Kumo (Japanese M)' },
  // Brazilian Portuguese
  { id: 'pf_dora', label: 'Dora (Portuguese F)' },
  { id: 'pm_alex', label: 'Alex (Portuguese M)' },
  { id: 'pm_santa', label: 'Santa (Portuguese M)' },
  // Mandarin Chinese
  { id: 'zf_xiaobei', label: 'Xiaobei (Chinese F)' },
  { id: 'zf_xiaoni', label: 'Xiaoni (Chinese F)' },
  { id: 'zf_xiaoxiao', label: 'Xiaoxiao (Chinese F)' },
  { id: 'zf_xiaoyi', label: 'Xiaoyi (Chinese F)' },
  { id: 'zm_yunjian', label: 'Yunjian (Chinese M)' },
  { id: 'zm_yunxi', label: 'Yunxi (Chinese M)' },
  { id: 'zm_yunxia', label: 'Yunxia (Chinese M)' },
  { id: 'zm_yunyang', label: 'Yunyang (Chinese M)' },
]

const CHATTERBOX_VOICES = [
  { id: 'default', label: 'Default' },
]

interface VoiceProfile {
  id: string
  workspace_id: string
  name: string
  provider: string
  voice_id: string
  reference_audio: string | null
  settings: Record<string, any>
  is_default: boolean
  created_at: string | null
  updated_at: string | null
}

// PRD-207 S7: the one status read behind the Auto Live card.
interface VoiceLiveStatus {
  platform_enabled: boolean
  armed: boolean
  workspace_enabled: boolean
  retell_voice_id: string | null
  used_minutes: number
  cap_minutes: number
  active_calls: number
  max_call_minutes: number
  viewer_is_admin: boolean
}

/**
 * Auto Live card (PRD-207 S7) — the workspace half of the two gates:
 * enable toggle, Retell voice id, the month's minute meter, and read-only
 * arming truth from the super-admin surface. Honest UI: the meter renders
 * real numbers from voice_calls, never placeholders.
 */
function AutoLiveCard() {
  const [status, setStatus] = useState<VoiceLiveStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [savingToggle, setSavingToggle] = useState(false)
  const [voiceId, setVoiceId] = useState('')
  const [savingVoice, setSavingVoice] = useState(false)
  const [armKey, setArmKey] = useState('')
  const [arming, setArming] = useState(false)

  const loadStatus = useCallback(async () => {
    try {
      const data = await apiClient.request<VoiceLiveStatus>('/api/voice/live-status')
      setStatus(data)
      setVoiceId(data.retell_voice_id || '')
    } catch (err) {
      console.error('Failed to load Auto Live status:', err)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    loadStatus()
  }, [loadStatus])

  const saveVoiceLive = async (patch: Record<string, unknown>) => {
    await apiClient.request('/api/workspaces/current/voice-live', {
      method: 'PUT',
      body: { voice_live: patch } as any,
    })
    await loadStatus()
  }

  const handleToggle = async (enabled: boolean) => {
    setSavingToggle(true)
    try {
      await saveVoiceLive({ enabled })
      toast.success(enabled ? 'Auto Live enabled for this workspace' : 'Auto Live disabled')
    } catch (err: any) {
      toast.error(err?.message || 'Failed to update Auto Live')
    } finally {
      setSavingToggle(false)
    }
  }

  const handleSaveVoice = async () => {
    const vid = voiceId.trim()
    if (vid.toLowerCase().startsWith('key_')) {
      toast.error("That's your Retell API key — it goes in the Arm box above, not the voice field")
      return
    }
    setSavingVoice(true)
    try {
      await saveVoiceLive({ retell_voice_id: vid })
      toast.success('Auto Live voice saved')
    } catch (err: any) {
      toast.error(err?.message || 'Failed to save the voice')
    } finally {
      setSavingVoice(false)
    }
  }

  // One-click arm: the server creates the Retell agent (right URLs, no
  // copy-paste) and files the key into the masked platform slots.
  const handleArm = async () => {
    const key = armKey.trim()
    if (!key) {
      toast.error('Paste your Retell API key first')
      return
    }
    setArming(true)
    try {
      const out = await apiClient.request<{ armed: boolean; agent_id?: string }>(
        '/api/voice/arm',
        { method: 'POST', body: { enabled: true, api_key: key } as any }
      )
      toast.success(`Auto Live armed — agent ${out.agent_id || 'ready'}`)
      setArmKey('')
      await loadStatus()
    } catch (err: any) {
      toast.error(err?.message || 'Arming failed')
    } finally {
      setArming(false)
    }
  }

  const handlePlatformToggle = async (enabled: boolean) => {
    setArming(true)
    try {
      await apiClient.request('/api/voice/arm', {
        method: 'POST',
        body: { enabled } as any,
      })
      toast.success(enabled ? 'Platform voice ON' : 'Platform voice OFF — all calls killed')
      await loadStatus()
    } catch (err: any) {
      toast.error(err?.message || 'Failed to switch platform voice')
    } finally {
      setArming(false)
    }
  }

  if (loading) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center py-8">
          <Loader2 className="w-5 h-5 animate-spin text-muted-foreground" />
        </CardContent>
      </Card>
    )
  }
  if (!status) return null

  const platformReady = status.platform_enabled && status.armed
  const overCap = status.used_minutes >= status.cap_minutes
  const meterPct = Math.min(100, Math.round((status.used_minutes / Math.max(1, status.cap_minutes)) * 100))

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Radio className="w-5 h-5" />
          Auto Live
          {status.workspace_enabled && platformReady ? (
            <Badge className="bg-warning/15 text-warning border-warning/30" variant="outline">
              live
            </Badge>
          ) : (
            <Badge variant="secondary">off</Badge>
          )}
        </CardTitle>
        <CardDescription className="mt-1">
          Real-time voice with Auto on the chat screen — interruptible, transcript in the
          thread, metered per workspace.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-5">
        {/* Platform truth + the controls that act on it, in one place */}
        <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
          <span>
            Platform voice:{' '}
            <span className={status.platform_enabled ? 'text-success' : 'text-destructive'}>
              {status.platform_enabled ? 'on' : 'off'}
            </span>
          </span>
          <span>·</span>
          <span>
            Retell:{' '}
            <span className={status.armed ? 'text-success' : 'text-warning'}>
              {status.armed ? 'armed' : 'not configured'}
            </span>
          </span>
          {!platformReady && !status.viewer_is_admin && (
            <span className="basis-full text-muted-foreground/80">
              An admin arms voice platform-wide right here.
            </span>
          )}
        </div>

        {/* One-click arming — paste the key, the server does the rest */}
        {status.viewer_is_admin && !status.armed && (
          <div className="rounded-xl border border-warning/30 bg-warning/5 px-4 py-3 space-y-2">
            <Label htmlFor="arm-key" className="font-medium">
              Arm Auto Live — paste your Retell API key
            </Label>
            <p className="text-xs text-muted-foreground">
              We create the &quot;Auto Live&quot; agent in your Retell account and wire
              every URL for you. No dashboard visit needed.
            </p>
            <div className="flex gap-2">
              <Input
                id="arm-key"
                type="password"
                value={armKey}
                onChange={(e) => setArmKey(e.target.value)}
                placeholder="key_…"
                className="font-mono"
                autoComplete="off"
              />
              <Button onClick={handleArm} disabled={arming}>
                {arming ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Arm'}
              </Button>
            </div>
          </div>
        )}

        {/* Armed: the master switch lives here too */}
        {status.viewer_is_admin && status.armed && (
          <div className="flex items-center justify-between rounded-xl border border-border/50 px-4 py-3">
            <div>
              <Label htmlFor="platform-voice" className="font-medium">
                Platform voice (master switch)
              </Label>
              <p className="text-xs text-muted-foreground mt-0.5">
                OFF refuses new calls and kills in-flight speech, platform-wide.
              </p>
            </div>
            <Switch
              id="platform-voice"
              checked={status.platform_enabled}
              onCheckedChange={handlePlatformToggle}
              disabled={arming}
            />
          </div>
        )}

        {/* The workspace gate */}
        <div className="flex items-center justify-between rounded-xl border border-border/50 px-4 py-3">
          <div>
            <Label htmlFor="auto-live-enabled" className="font-medium">
              Enable Auto Live in this workspace
            </Label>
            <p className="text-xs text-muted-foreground mt-0.5">
              Members get the Live button on the chat screen.
            </p>
          </div>
          <Switch
            id="auto-live-enabled"
            checked={status.workspace_enabled}
            onCheckedChange={handleToggle}
            disabled={savingToggle}
          />
        </div>

        {/* Voice (per-call override rides the mint — verified supported) */}
        <div>
          <Label htmlFor="auto-live-voice">Auto&apos;s voice (Retell voice id)</Label>
          <div className="mt-1 flex gap-2">
            <Input
              id="auto-live-voice"
              value={voiceId}
              onChange={(e) => setVoiceId(e.target.value)}
              placeholder="e.g. 11labs-Adrian — empty = the Retell agent's default"
              className="font-mono"
            />
            <Button size="sm" onClick={handleSaveVoice} disabled={savingVoice}>
              {savingVoice ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Save'}
            </Button>
          </div>
          <p className="text-xs text-muted-foreground mt-1">
            Any voice id from your Retell dashboard; applied per call.
          </p>
        </div>

        {/* The month's meter — real numbers from voice_calls */}
        <div>
          <div className="flex items-center justify-between text-sm">
            <span className="font-medium">This month</span>
            <span className={overCap ? 'text-warning font-medium' : 'text-muted-foreground'}>
              {status.used_minutes}/{status.cap_minutes} min
              {status.active_calls > 0 && ` · ${status.active_calls} live now`}
            </span>
          </div>
          <Progress value={meterPct} className="mt-2 h-2" />
          {overCap && (
            <p className="text-xs text-warning mt-1">
              Voice budget used — new calls are refused until next month (or raise the cap).
            </p>
          )}
        </div>
      </CardContent>
    </Card>
  )
}

export function VoiceProfilesSettingsTab() {
  const [profiles, setProfiles] = useState<VoiceProfile[]>([])
  const [loading, setLoading] = useState(true)
  const [creating, setCreating] = useState(false)
  const [deleting, setDeleting] = useState<string | null>(null)
  const [previewing, setPreviewing] = useState<string | null>(null)

  // Create form state
  const [showCreate, setShowCreate] = useState(false)
  const [newName, setNewName] = useState('')
  const [newProvider, setNewProvider] = useState('kokoro')
  const [newVoiceId, setNewVoiceId] = useState('')
  const [newIsDefault, setNewIsDefault] = useState(false)

  // Clone form state
  const [showClone, setShowClone] = useState(false)
  const [cloneName, setCloneName] = useState('')
  const [cloneFile, setCloneFile] = useState<File | null>(null)
  const [cloning, setCloning] = useState(false)

  const loadProfiles = useCallback(async () => {
    try {
      const data = await apiClient.request<{ items: VoiceProfile[]; total: number }>(
        '/api/voice/profiles'
      )
      setProfiles(data.items || [])
    } catch (err) {
      console.error('Failed to load voice profiles:', err)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    loadProfiles()
  }, [loadProfiles])

  const handleCreate = async () => {
    if (!newName.trim() || !newVoiceId) return
    setCreating(true)
    try {
      await apiClient.request('/api/voice/profiles', {
        method: 'POST',
        body: {
          name: newName.trim(),
          provider: newProvider,
          voice_id: newVoiceId,
          is_default: newIsDefault,
        } as any,
      })
      toast.success(`Voice profile "${newName}" created`)
      setShowCreate(false)
      setNewName('')
      setNewVoiceId('')
      setNewIsDefault(false)
      await loadProfiles()
    } catch (err: any) {
      toast.error(err?.message || 'Failed to create profile')
    } finally {
      setCreating(false)
    }
  }

  const handleClone = async () => {
    if (!cloneName.trim() || !cloneFile) return
    setCloning(true)
    try {
      const formData = new FormData()
      formData.append('file', cloneFile)

      const headers = await apiClient.getAuthHeaders()
      const params = new URLSearchParams({ name: cloneName.trim(), provider: 'chatterbox' })
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL || ''}/api/voice/profiles/clone?${params}`,
        { method: 'POST', headers, body: formData }
      )

      if (!response.ok) {
        const errText = await response.text().catch(() => '')
        throw new Error(errText || `Clone failed (${response.status})`)
      }

      toast.success(`Cloned voice "${cloneName}" created`)
      setShowClone(false)
      setCloneName('')
      setCloneFile(null)
      await loadProfiles()
    } catch (err: any) {
      toast.error(err?.message || 'Voice cloning failed')
    } finally {
      setCloning(false)
    }
  }

  const handleDelete = async (profile: VoiceProfile) => {
    setDeleting(profile.id)
    try {
      await apiClient.request(`/api/voice/profiles/${profile.id}`, { method: 'DELETE' })
      toast.success(`Deleted "${profile.name}"`)
      await loadProfiles()
    } catch (err: any) {
      toast.error(err?.message || 'Failed to delete')
    } finally {
      setDeleting(null)
    }
  }

  const handlePreview = async (profile: VoiceProfile) => {
    if (previewing) return // Already playing a preview

    setPreviewing(profile.id)
    try {
      const headers = await apiClient.getAuthHeaders()
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL || ''}/api/voice/profiles/${profile.id}/preview`,
        { method: 'POST', headers }
      )
      if (!response.ok) {
        throw new Error(`Preview failed (${response.status})`)
      }
      const data = await response.json()

      // Decode base64 to raw bytes
      const raw = atob(data.audio_base64)
      const bytes = new Uint8Array(raw.length)
      for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i)

      // Use AudioContext for reliable playback (avoids autoplay restrictions)
      const ctx = new AudioContext()
      const buffer = await ctx.decodeAudioData(bytes.buffer)
      const source = ctx.createBufferSource()
      source.buffer = buffer
      source.connect(ctx.destination)

      source.onended = () => {
        setPreviewing(null)
        ctx.close()
      }

      source.start(0)
      toast.success(`Playing ${profile.name}...`, { duration: 2000 })
    } catch (err: any) {
      toast.error(err?.message || 'Preview failed')
      setPreviewing(null)
    }
  }

  const voiceOptions = newProvider === 'kokoro' ? KOKORO_VOICES : CHATTERBOX_VOICES

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* PRD-207 S7: Auto Live — the workspace gate, voice, meter */}
      <AutoLiveCard />

      {/* Header */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Volume2 className="w-5 h-5" />
                Voice Profiles
              </CardTitle>
              <CardDescription className="mt-1">
                Configure TTS voices for your agents. Assign profiles to agents for personalized voice responses.
              </CardDescription>
            </div>
            <div className="flex gap-2">
              <Dialog open={showClone} onOpenChange={setShowClone}>
                <DialogTrigger asChild>
                  <Button variant="outline" size="sm">
                    <Mic className="w-4 h-4 mr-1" />
                    Clone Voice
                  </Button>
                </DialogTrigger>
                <DialogContent>
                  <DialogHeader>
                    <DialogTitle>Clone a Voice</DialogTitle>
                    <DialogDescription>
                      Upload 5-60 seconds of reference audio to create a cloned voice profile using Chatterbox.
                    </DialogDescription>
                  </DialogHeader>
                  <div className="space-y-4 pt-2">
                    <div>
                      <Label>Profile Name</Label>
                      <Input
                        value={cloneName}
                        onChange={(e) => setCloneName(e.target.value)}
                        placeholder="e.g. My Custom Voice"
                        className="mt-1"
                      />
                    </div>
                    <div>
                      <Label>Reference Audio</Label>
                      <div className="mt-1 flex items-center gap-3">
                        <Input
                          type="file"
                          accept=".wav,.mp3,.webm"
                          onChange={(e) => setCloneFile(e.target.files?.[0] || null)}
                          className="flex-1"
                        />
                      </div>
                      <p className="text-xs text-muted-foreground mt-1">
                        WAV, MP3, or WebM. 5-60 seconds of clear speech.
                      </p>
                    </div>
                    <Button
                      onClick={handleClone}
                      disabled={!cloneName.trim() || !cloneFile || cloning}
                      className="w-full"
                    >
                      {cloning ? (
                        <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Cloning...</>
                      ) : (
                        <><Upload className="w-4 h-4 mr-2" /> Create Cloned Voice</>
                      )}
                    </Button>
                  </div>
                </DialogContent>
              </Dialog>

              <Dialog open={showCreate} onOpenChange={setShowCreate}>
                <DialogTrigger asChild>
                  <Button size="sm">
                    <Plus className="w-4 h-4 mr-1" />
                    New Profile
                  </Button>
                </DialogTrigger>
                <DialogContent>
                  <DialogHeader>
                    <DialogTitle>Create Voice Profile</DialogTitle>
                    <DialogDescription>
                      Create a new voice profile from built-in voices.
                    </DialogDescription>
                  </DialogHeader>
                  <div className="space-y-4 pt-2">
                    <div>
                      <Label>Profile Name</Label>
                      <Input
                        value={newName}
                        onChange={(e) => setNewName(e.target.value)}
                        placeholder="e.g. Friendly Agent Voice"
                        className="mt-1"
                      />
                    </div>
                    <div>
                      <Label>Provider</Label>
                      <Select value={newProvider} onValueChange={(v) => { setNewProvider(v); setNewVoiceId('') }}>
                        <SelectTrigger className="mt-1">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="kokoro">Kokoro (Fast)</SelectItem>
                          <SelectItem value="chatterbox">Chatterbox (High Quality)</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div>
                      <Label>Voice</Label>
                      <Select value={newVoiceId} onValueChange={setNewVoiceId}>
                        <SelectTrigger className="mt-1">
                          <SelectValue placeholder="Select a voice" />
                        </SelectTrigger>
                        <SelectContent>
                          {voiceOptions.map((v) => (
                            <SelectItem key={v.id} value={v.id}>{v.label}</SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        id="new-default"
                        checked={newIsDefault}
                        onChange={(e) => setNewIsDefault(e.target.checked)}
                        className="rounded"
                      />
                      <Label htmlFor="new-default" className="text-sm font-normal">
                        Set as workspace default
                      </Label>
                    </div>
                    <Button
                      onClick={handleCreate}
                      disabled={!newName.trim() || !newVoiceId || creating}
                      className="w-full"
                    >
                      {creating ? (
                        <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Creating...</>
                      ) : (
                        'Create Profile'
                      )}
                    </Button>
                  </div>
                </DialogContent>
              </Dialog>
            </div>
          </div>
        </CardHeader>

        <CardContent>
          {profiles.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">
              <Volume2 className="w-10 h-10 mx-auto mb-3 opacity-40" />
              <p className="text-sm">No voice profiles yet. Create one to get started.</p>
            </div>
          ) : (
            <div className="space-y-3">
              {profiles.map((profile) => (
                <div
                  key={profile.id}
                  className="flex items-center justify-between rounded-xl border border-border/50 bg-card/50 px-4 py-3 transition-colors hover:bg-accent/30"
                >
                  <div className="flex items-center gap-3 min-w-0">
                    <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary">
                      {profile.reference_audio ? (
                        <Mic className="w-4 h-4" />
                      ) : (
                        <Volume2 className="w-4 h-4" />
                      )}
                    </div>
                    <div className="min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="font-medium text-sm truncate">{profile.name}</span>
                        {profile.is_default && (
                          <Badge variant="secondary" className="text-[10px] px-1.5 py-0">
                            <Star className="w-3 h-3 mr-0.5" /> Default
                          </Badge>
                        )}
                        {profile.reference_audio && (
                          <Badge variant="outline" className="text-[10px] px-1.5 py-0">
                            Cloned
                          </Badge>
                        )}
                      </div>
                      <p className="text-xs text-muted-foreground">
                        {profile.provider} / {profile.voice_id}
                      </p>
                    </div>
                  </div>

                  <div className="flex items-center gap-1 shrink-0">
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8"
                      onClick={() => handlePreview(profile)}
                      disabled={previewing !== null && previewing !== profile.id}
                    >
                      {previewing === profile.id ? (
                        <Square className="w-3.5 h-3.5" />
                      ) : (
                        <Play className="w-3.5 h-3.5" />
                      )}
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8 text-destructive hover:text-destructive"
                      onClick={() => handleDelete(profile)}
                      disabled={deleting === profile.id}
                    >
                      {deleting === profile.id ? (
                        <Loader2 className="w-3.5 h-3.5 animate-spin" />
                      ) : (
                        <Trash2 className="w-3.5 h-3.5" />
                      )}
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
