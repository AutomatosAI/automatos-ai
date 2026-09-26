'use client'

/**
 * PRD-251 S1.5 (D11, D15) — the voice a post is spoken with.
 *
 * Kokoro, built into the renderer and free, is always offered. A voice toolkit
 * the workspace has connected in Composio (Fish Audio, ElevenLabs) is offered
 * beside it, and choosing one lists its voices. One the workspace has not
 * connected shows a Connect button that runs the Composio connect flow (the
 * Tools page's own: POST /api/composio/connect/{app} and its hosted sign-in);
 * it becomes a choice once connected. The choice is saved on the post and used
 * by its next render. It is a render setting, so it never voids an approval.
 */
import { useEffect, useMemo, useState, type FormEvent } from 'react'
import { Loader2, Plug } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import type { SocialPost, SocialPostVoice, SocialVoiceSource } from '@/lib/api-client'
import { debounce } from '@/lib/utils'
import { useInitiateConnection } from '@/hooks/use-composio-api'
import { useSocialToolkitVoices, useSocialVoiceSources, useUpdateSocialPost } from '@/hooks/use-socials-api'

/** The built-in voice's toolkit name (modules/socials/service.py KOKORO). */
export const KOKORO = 'kokoro'
const KOKORO_FALLBACK_LABEL = 'Kokoro (built in)'
// What the Composio callback page posts to the window that opened it.
const CONNECTED_MESSAGE = 'COMPOSIO_CONNECTED'
const POPUP_FEATURES = 'width=600,height=700'
const SELECT_CLASS = 'h-9 w-full rounded-md border border-input bg-background px-2 text-sm'
// A search asks the toolkit (a Composio call) once typing pauses, not per keystroke.
export const VOICE_SEARCH_DEBOUNCE_MS = 300

interface SocialsVoicePickerProps {
  post: SocialPost
  /** Whether the caller may change the voice: an author, on a post that can be edited. */
  editable: boolean
}

/** "Fish Audio — Energetic narrator", or Kokoro's label. */
export function voiceLabel(voice: SocialPostVoice | null | undefined, sources: SocialVoiceSource[]): string {
  if (!voice) return sources.find((s) => s.toolkit === KOKORO)?.label ?? KOKORO_FALLBACK_LABEL
  const label = sources.find((s) => s.toolkit === voice.toolkit)?.label ?? voice.toolkit
  return `${label} — ${voice.name || voice.voice_id}`
}

function ConnectToolkit({ source }: { source: SocialVoiceSource }) {
  const initiate = useInitiateConnection()
  const [failed, setFailed] = useState(false)

  const connect = async () => {
    setFailed(false)
    const appName = source.toolkit.toUpperCase()
    try {
      const result = await initiate.mutateAsync({
        appName,
        callbackUrl: `${window.location.origin}/tools/callback?connected=${appName}`,
      })
      if (result?.redirect_url) window.open(result.redirect_url, `Connect ${source.label}`, POPUP_FEATURES)
    } catch {
      setFailed(true)
    }
  }

  return (
    <div className="flex flex-wrap items-center gap-2">
      <Button type="button" size="sm" variant="outline" onClick={connect} disabled={initiate.isLoading}>
        <Plug className="mr-1.5 h-4 w-4" aria-hidden />
        Connect {source.label}
      </Button>
      <span className="text-xs text-muted-foreground">Connect it in Composio to use your {source.label} voices.</span>
      {failed && (
        <span role="alert" className="text-xs text-destructive">
          Could not start the {source.label} connection. Try again.
        </span>
      )}
    </div>
  )
}

export function SocialsVoicePicker({ post, editable }: SocialsVoicePickerProps) {
  const sources = useSocialVoiceSources()
  const update = useUpdateSocialPost()
  const saved = post.voice?.toolkit ?? KOKORO
  const [toolkit, setToolkit] = useState(saved)
  const [query, setQuery] = useState('')
  const [searched, setSearched] = useState('')
  const [typedId, setTypedId] = useState('')
  const searchLater = useMemo(() => debounce((value: string) => setSearched(value), VOICE_SEARCH_DEBOUNCE_MS), [])

  // A refetch (after a save) carries the server's voice; follow it.
  useEffect(() => {
    setToolkit(saved)
    setQuery('')
    setSearched('')
    setTypedId('')
  }, [post.id, saved])

  // The connect flow ends in its own window: read the choices again when it reports back.
  const { refetch } = sources
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

  const all = sources.data?.sources ?? []
  const choices = all.filter((s) => s.status === 'available')
  const chosen = choices.find((s) => s.toolkit === toolkit)
  const listing = chosen && !chosen.builtin && chosen.lists_voices ? chosen.toolkit : null
  const voices = useSocialToolkitVoices(listing, searched)
  const busy = update.isLoading

  if (!editable) {
    return <p className="text-sm text-muted-foreground">Voice: {voiceLabel(post.voice, all)}</p>
  }

  const save = (voice: SocialPostVoice | null) => update.mutate({ postId: post.id, changes: { voice } })

  const choose = (next: string) => {
    setToolkit(next)
    if (next === KOKORO && saved !== KOKORO) save(null)
  }

  const pick = (voiceId: string) => {
    const voice = voices.data?.voices.find((v) => v.id === voiceId)
    if (voice) save({ toolkit, voice_id: voice.id, name: voice.name })
  }

  const submitTypedId = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (typedId.trim()) save({ toolkit, voice_id: typedId.trim() })
  }

  const savedId = post.voice?.toolkit === toolkit ? post.voice.voice_id : ''
  const listed = voices.data?.voices ?? []

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
              <input
                type="radio"
                name={`socials-voice-${post.id}`}
                value={source.toolkit}
                checked={toolkit === source.toolkit}
                onChange={() => choose(source.toolkit)}
                disabled={busy}
              />
              {source.label}
            </label>
          ))}
        </div>
      )}
      <p className="text-xs text-muted-foreground">
        Kokoro is free and built into the renderer. A voice toolkit speaks through your own account, connected in Composio.
      </p>

      {chosen && !chosen.builtin && listing && (
        <div className="space-y-1.5">
          <Input
            aria-label={`Search ${chosen.label} voices`}
            value={query}
            onChange={(event) => {
              setQuery(event.target.value)
              searchLater(event.target.value)
            }}
            placeholder="Search voices by name"
          />
          <select
            aria-label={`${chosen.label} voice`}
            value={savedId}
            onChange={(event) => pick(event.target.value)}
            disabled={busy || voices.isLoading}
            className={SELECT_CLASS}
          >
            <option value="" disabled>
              {voices.isLoading ? 'Loading voices…' : 'Choose a voice'}
            </option>
            {savedId && !listed.some((v) => v.id === savedId) && (
              <option value={savedId}>{post.voice?.name || savedId}</option>
            )}
            {listed.map((voice) => (
              <option key={voice.id} value={voice.id}>
                {voice.name}
              </option>
            ))}
          </select>
          {voices.isError && (
            <p role="alert" className="text-xs text-destructive">
              {voices.error instanceof Error ? voices.error.message : `Could not list the ${chosen.label} voices.`}
            </p>
          )}
        </div>
      )}

      {chosen && !chosen.builtin && !listing && (
        <form onSubmit={submitTypedId} aria-label={`${chosen.label} voice id`} className="flex gap-2">
          <Input
            aria-label={`${chosen.label} voice id`}
            value={typedId}
            onChange={(event) => setTypedId(event.target.value)}
            placeholder={savedId || 'The voice id from your account'}
          />
          <Button type="submit" size="sm" variant="outline" disabled={busy || !typedId.trim()}>
            Use this voice
          </Button>
        </form>
      )}

      {all
        .filter((source) => source.status === 'connect')
        .map((source) => (
          <ConnectToolkit key={source.toolkit} source={source} />
        ))}
      {all
        .filter((source) => source.status === 'unavailable')
        .map((source) => (
          <p key={source.toolkit} className="text-xs text-muted-foreground">
            {source.label}: {source.reason}
          </p>
        ))}
      {sources.isError && (
        <p className="text-xs text-muted-foreground">Could not load the voices. Kokoro still speaks this post.</p>
      )}
    </fieldset>
  )
}
