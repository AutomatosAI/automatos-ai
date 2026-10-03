'use client'

/**
 * PRD-251 S1.5 (D11, D15) — the voice a post is spoken with.
 *
 * Kokoro, built into the renderer and free, is always offered: the template's own voice,
 * or one of Kokoro's own voices from its catalogue (PRD-251B US-B306). A voice toolkit
 * the workspace has connected in Composio (Fish Audio, ElevenLabs) is offered beside it,
 * and choosing one lists its voices. One the workspace has not connected shows a Connect
 * button that runs the Composio connect flow; it becomes a choice once connected. The
 * choice is saved on the post and used by its next render. It is a render setting, so it
 * never voids an approval.
 */
import { useEffect, useMemo, useState, type FormEvent } from 'react'
import { Loader2 } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import type { SocialPost, SocialPostVoice, SocialToolkitVoice, SocialVoiceSource } from '@/lib/api-client'
import { debounce } from '@/lib/utils'
import { useSocialToolkitVoices, useSocialVoiceSources, useUpdateSocialPost } from '@/hooks/use-socials-api'
import { CONNECTED_MESSAGE, ConnectToolkit } from './connect-toolkit'

/** The built-in voice's toolkit name (modules/socials/service.py KOKORO). */
export const KOKORO = 'kokoro'
const KOKORO_FALLBACK_LABEL = 'Kokoro (built in)'
/** Kokoro's choice that keeps the template's own voice (af_heart in every starter). */
export const TEMPLATE_VOICE = "The template's own voice"
const SELECT_CLASS = 'h-9 w-full rounded-md border border-input bg-background px-2 text-sm'
// A search asks the toolkit (a Composio call) once typing pauses, not per keystroke.
export const VOICE_SEARCH_DEBOUNCE_MS = 300

interface SocialsVoicePickerProps {
  post: SocialPost
  /** Whether the caller may change the voice: an author, on a post that can be edited. */
  editable: boolean
}

/** "Fish Audio — Energetic narrator", "Kokoro (built in) — George", or Kokoro's label. */
export function voiceLabel(voice: SocialPostVoice | null | undefined, sources: SocialVoiceSource[]): string {
  if (!voice) return sources.find((s) => s.toolkit === KOKORO)?.label ?? KOKORO_FALLBACK_LABEL
  const label = sources.find((s) => s.toolkit === voice.toolkit)?.label ?? voice.toolkit
  return `${label} — ${voice.name || voice.voice_id}`
}

interface VoiceListProps {
  source: SocialVoiceSource
  saved: SocialPostVoice | null | undefined
  busy: boolean
  onPick: (voice: SocialToolkitVoice | null) => void
}

/** A source's voices, searchable; Kokoro's list starts with the template's own voice. */
function VoiceList({ source, saved, busy, onPick }: VoiceListProps) {
  const [query, setQuery] = useState('')
  const [searched, setSearched] = useState('')
  const searchLater = useMemo(() => debounce((value: string) => setSearched(value), VOICE_SEARCH_DEBOUNCE_MS), [])
  const voices = useSocialToolkitVoices(source.toolkit, searched)
  const listed = voices.data?.voices ?? []
  const savedId = saved?.toolkit === source.toolkit ? saved.voice_id : ''
  const pick = (id: string) => onPick(id ? listed.find((v) => v.id === id) ?? null : null)
  return (
    <div className="space-y-1.5">
      <Input
        aria-label={`Search ${source.label} voices`}
        value={query}
        onChange={(event) => {
          setQuery(event.target.value)
          searchLater(event.target.value)
        }}
        placeholder="Search voices by name"
      />
      <select aria-label={`${source.label} voice`} value={savedId} onChange={(event) => pick(event.target.value)} disabled={busy || voices.isLoading} className={SELECT_CLASS}>
        <option value="" disabled={!source.builtin}>
          {source.builtin ? TEMPLATE_VOICE : voices.isLoading ? 'Loading voices…' : 'Choose a voice'}
        </option>
        {savedId && !listed.some((v) => v.id === savedId) && <option value={savedId}>{saved?.name || savedId}</option>}
        {listed.map((voice) => (
          <option key={voice.id} value={voice.id}>
            {voice.description ? `${voice.name} — ${voice.description}` : voice.name}
          </option>
        ))}
      </select>
      {voices.isError && (
        <p role="alert" className="text-xs text-destructive">
          {voices.error instanceof Error ? voices.error.message : `Could not list the ${source.label} voices.`}
        </p>
      )}
    </div>
  )
}

function TypedVoiceId({ source, placeholder, busy, onUse }: { source: SocialVoiceSource; placeholder: string; busy: boolean; onUse: (id: string) => void }) {
  const [typedId, setTypedId] = useState('')
  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (typedId.trim()) onUse(typedId.trim())
  }
  return (
    <form onSubmit={submit} aria-label={`${source.label} voice id`} className="flex gap-2">
      <Input aria-label={`${source.label} voice id`} value={typedId} onChange={(event) => setTypedId(event.target.value)} placeholder={placeholder || 'The voice id from your account'} />
      <Button type="submit" size="sm" variant="outline" disabled={busy || !typedId.trim()}>
        Use this voice
      </Button>
    </form>
  )
}

/** The connect flow ends in its own window: read the choices again when it reports back. */
function useRefetchOnConnect(refetch: () => unknown) {
  useEffect(() => {
    const onMessage = (event: MessageEvent) => {
      if (event.origin === window.location.origin && event.data?.type === CONNECTED_MESSAGE) void refetch()
    }
    const onFocus = () => void refetch()
    window.addEventListener('message', onMessage)
    window.addEventListener('focus', onFocus)
    return () => {
      window.removeEventListener('message', onMessage)
      window.removeEventListener('focus', onFocus)
    }
  }, [refetch])
}

export function SocialsVoicePicker({ post, editable }: SocialsVoicePickerProps) {
  const sources = useSocialVoiceSources()
  const update = useUpdateSocialPost()
  const saved = post.voice?.toolkit ?? KOKORO
  const [toolkit, setToolkit] = useState(saved)
  // A refetch (after a save) carries the server's voice; follow it.
  useEffect(() => setToolkit(saved), [post.id, saved])
  useRefetchOnConnect(sources.refetch)

  const all = sources.data?.sources ?? []
  const choices = all.filter((s) => s.status === 'available')
  const chosen = choices.find((s) => s.toolkit === toolkit)
  const busy = update.isLoading

  if (!editable) {
    return <p className="text-sm text-muted-foreground">Voice: {voiceLabel(post.voice, all)}</p>
  }

  const save = (voice: SocialPostVoice | null) => update.mutate({ postId: post.id, changes: { voice } })
  const choose = (next: string) => {
    setToolkit(next)
    if (next === KOKORO && saved !== KOKORO) save(null)
  }
  const pick = (voice: SocialToolkitVoice | null) => save(voice ? { toolkit, voice_id: voice.id, name: voice.name } : null)

  return (
    <fieldset className="space-y-2">
      <legend className="text-sm font-medium text-foreground">Voice</legend>
      {sources.isLoading ? (
        <p className="flex items-center gap-2 text-sm text-muted-foreground">
          <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
          Loading voices…
        </p>
      ) : (
        <div className="flex flex-wrap gap-x-4 gap-y-1.5">
          {choices.map((source) => (
            <label key={source.toolkit} className="flex items-center gap-2 text-sm text-foreground">
              <input type="radio" name={`socials-voice-${post.id}`} value={source.toolkit} checked={toolkit === source.toolkit} onChange={() => choose(source.toolkit)} disabled={busy} />
              {source.label}
            </label>
          ))}
        </div>
      )}
      <p className="text-xs text-muted-foreground">
        Kokoro is free and built into the renderer. A voice toolkit speaks through your own account, connected in Composio.
      </p>
      {chosen && chosen.lists_voices && <VoiceList key={`${post.id}-${chosen.toolkit}`} source={chosen} saved={post.voice} busy={busy} onPick={pick} />}
      {chosen && !chosen.builtin && !chosen.lists_voices && (
        <TypedVoiceId key={`${post.id}-${chosen.toolkit}`} source={chosen} busy={busy} placeholder={post.voice?.toolkit === toolkit ? post.voice.voice_id : ''} onUse={(id) => save({ toolkit, voice_id: id })} />
      )}
      {all.filter((source) => source.status === 'connect').map((source) => (
        <ConnectToolkit key={source.toolkit} toolkit={source.toolkit} label={source.label} purpose={`your ${source.label} voices`} />
      ))}
      {all.filter((source) => source.status === 'unavailable').map((source) => (
        <p key={source.toolkit} className="text-xs text-muted-foreground">
          {source.label}: {source.reason}
        </p>
      ))}
      {sources.isError && <p className="text-xs text-muted-foreground">Could not load the voices. Kokoro still speaks this post.</p>}
    </fieldset>
  )
}
