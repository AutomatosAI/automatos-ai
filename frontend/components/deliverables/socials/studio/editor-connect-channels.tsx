'use client'

/**
 * PRD-251D (E5, US-D105) — connect a channel from the editor: each supported Socials channel
 * (the brand kit's list, keyed by Composio toolkit) that the workspace has not connected gets
 * the Composio connect button (connect-toolkit.tsx, the flow the voice picker uses). The
 * channels route lists the connected channels only, so the rest are the supported ones it does
 * not name. When the connect window reports back, or the page gets the focus again, the
 * channels are read again, so a new channel becomes a row to tick.
 */
import type { SocialChannel } from '@/lib/api-client'
import { SOCIAL_HANDLE_FIELDS } from '@/components/documents/blocks/BrandKitSocial'
import { ConnectToolkit, useRefetchOnConnect } from '../connect-toolkit'
import { channelLabel } from '../socials-status'

/** The supported Socials channels, by Composio toolkit, in the brand kit's order. */
export const SUPPORTED_CHANNELS: ReadonlyArray<string> = SOCIAL_HANDLE_FIELDS.map((field) => field.toolkit)

/** "X for your posts": what connecting a channel brings (ConnectToolkit's `purpose`). */
export function connectPurpose(toolkit: string): string {
  return `${channelLabel(toolkit)} for your posts`
}

/** The supported channels the workspace has not connected, in the brand kit's order. */
export function unconnectedChannels(channels: ReadonlyArray<Pick<SocialChannel, 'toolkit'>>): string[] {
  const connected = new Set(channels.map((channel) => channel.toolkit))
  return SUPPORTED_CHANNELS.filter((toolkit) => !connected.has(toolkit))
}

interface EditorConnectChannelsProps {
  channels: ReadonlyArray<SocialChannel>
  /** Reads the connected channels again. */
  refetch: () => unknown
}

export function EditorConnectChannels({ channels, refetch }: EditorConnectChannelsProps) {
  useRefetchOnConnect(refetch)
  const missing = unconnectedChannels(channels)
  if (missing.length === 0) return null
  return (
    <ul aria-label="Channels to connect" className="flex flex-col gap-2">
      {missing.map((toolkit) => (
        <li key={toolkit}>
          <ConnectToolkit toolkit={toolkit} label={channelLabel(toolkit)} purpose={connectPurpose(toolkit)} />
        </li>
      ))}
    </ul>
  )
}
