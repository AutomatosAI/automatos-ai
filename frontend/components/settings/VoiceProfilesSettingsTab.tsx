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
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Badge } from '@/components/ui/badge'
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

// Known voices per provider
const KOKORO_VOICES = [
  { id: 'af_heart', label: 'Heart (Female)' },
  { id: 'af_bella', label: 'Bella (Female)' },
  { id: 'af_nicole', label: 'Nicole (Female)' },
  { id: 'af_sarah', label: 'Sarah (Female)' },
  { id: 'af_sky', label: 'Sky (Female)' },
  { id: 'am_adam', label: 'Adam (Male)' },
  { id: 'am_michael', label: 'Michael (Male)' },
  { id: 'bf_emma', label: 'Emma (British F)' },
  { id: 'bm_george', label: 'George (British M)' },
  { id: 'bm_lewis', label: 'Lewis (British M)' },
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

export function VoiceProfilesSettingsTab() {
  const [profiles, setProfiles] = useState<VoiceProfile[]>([])
  const [loading, setLoading] = useState(true)
  const [creating, setCreating] = useState(false)
  const [deleting, setDeleting] = useState<string | null>(null)
  const [previewing, setPreviewing] = useState<string | null>(null)
  const [previewAudio, setPreviewAudio] = useState<HTMLAudioElement | null>(null)

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
    // Stop any current preview
    if (previewAudio) {
      previewAudio.pause()
      setPreviewAudio(null)
      if (previewing === profile.id) {
        setPreviewing(null)
        return
      }
    }

    setPreviewing(profile.id)
    try {
      const data = await apiClient.request<{
        audio_base64: string
        format: string
        duration_ms: number
      }>(`/api/voice/profiles/${profile.id}/preview`, { method: 'POST' })

      const audioBytes = Uint8Array.from(atob(data.audio_base64), (c) => c.charCodeAt(0))
      const blob = new Blob([audioBytes], { type: 'audio/mpeg' })
      const url = URL.createObjectURL(blob)
      const audio = new Audio(url)

      audio.onended = () => {
        setPreviewing(null)
        setPreviewAudio(null)
        URL.revokeObjectURL(url)
      }
      audio.onerror = () => {
        setPreviewing(null)
        setPreviewAudio(null)
        URL.revokeObjectURL(url)
      }

      setPreviewAudio(audio)
      await audio.play()
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
                    <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-orange-500/10 text-orange-500">
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
