/**
 * PRD-261 A-12a — SDK area `deliverables`. The hooks are the host's own (so every read carries the
 * workspace and the token through the backend client), given the theme's arguments as a copy the
 * host may keep; the filters and the tab labels are frozen copies of the host's; the URL and name
 * helpers are the host's own functions; a kind's label is the host's, and a kind it has no name for
 * is named as it is. The host's hooks are stood in for here, recording what they were given.
 */
import { renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

const host = vi.hoisted(() => ({
  calls: [] as { readonly hook: string; readonly args: readonly unknown[] }[],
  listResult: { data: undefined, isLoading: true } as Record<string, unknown>,
  detailResult: { data: undefined, isLoading: true } as Record<string, unknown>,
  download: { download: async () => undefined, downloading: false },
  blob: { src: 'blob:a12a', error: null },
}))

vi.mock('@/hooks/use-deliverables-api', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@/hooks/use-deliverables-api')>()),
  useDeliverables: (...args: unknown[]) => {
    host.calls.push({ hook: 'useDeliverables', args })
    return host.listResult
  },
  useDeliverable: (...args: unknown[]) => {
    host.calls.push({ hook: 'useDeliverable', args })
    return host.detailResult
  },
}))
vi.mock('@/hooks/use-api-file-download', () => ({
  useApiFileDownload: (...args: unknown[]) => {
    host.calls.push({ hook: 'useApiFileDownload', args })
    return host.download
  },
}))
vi.mock('@/components/widgets/FileWidget/FilePreview', () => ({
  useAuthenticatedBlobUrl: (...args: unknown[]) => {
    host.calls.push({ hook: 'useAuthenticatedBlobUrl', args })
    return host.blob
  },
}))

import {
  DELIVERABLE_FEED_FILTERS,
  DELIVERABLE_TAB_LABELS,
  deliverableDownloadUrl,
  deliverableFilename,
  deliverableFileUrl,
  deliverableLabel,
  fileDownloadUrl,
  useAuthenticatedBlobUrl,
  useDeliverable,
  useDeliverables,
  useFileDownload,
  visibleDeliverableTabs,
  type DeliverableFilters,
} from '../deliverables'
import { deliverableLabel as hostLabel } from '@/components/icons/deliverable-icon'
import { DEFAULT_FILTERS, FEED_DEFAULT_FILTERS } from '@/hooks/use-deliverables-api'
import * as download from '@/lib/deliverables/download'
import { TAB_LABELS, visibleDeliverableTabs as hostVisibleTabs } from '@/lib/deliverables/tabs'

function lastCall(hook: string): readonly unknown[] {
  const call = [...host.calls].reverse().find((entry) => entry.hook === hook)
  if (call === undefined) throw new Error(`${hook} was not called`)
  return call.args
}

afterEach(() => {
  host.calls.length = 0
})

describe('PRD-261 A-12a — the Deliverables hooks', () => {
  it("useDeliverables reads the host's list with a copy of the theme's filters and its options", () => {
    const filters: DeliverableFilters = { ...DELIVERABLE_FEED_FILTERS, artifact_type: 'report', tag: 'launch' }
    const { result } = renderHook(() => useDeliverables(filters, { refetchInterval: 30_000 }))
    const [given, options] = lastCall('useDeliverables')
    expect(given).toEqual(filters)
    expect(given).not.toBe(filters)
    expect((given as DeliverableFilters).source_type_exclude).not.toBe(filters.source_type_exclude)
    expect(Object.isFrozen((given as DeliverableFilters).source_type_exclude)).toBe(false)
    expect(options).toEqual({ refetchInterval: 30_000 })
    expect(result.current).toBe(host.listResult)
  })

  it('useDeliverables reads every Deliverable when the theme gives no filters', () => {
    renderHook(() => useDeliverables())
    expect(lastCall('useDeliverables')[0]).toEqual(DEFAULT_FILTERS)
  })

  it("useDeliverable reads one Deliverable, its content only when asked", () => {
    const { result } = renderHook(() => useDeliverable('d-1', true))
    expect(lastCall('useDeliverable')).toEqual(['d-1', true])
    expect(result.current).toBe(host.detailResult)
    renderHook(() => useDeliverable(null))
    expect(lastCall('useDeliverable')).toEqual([null, false])
  })

  it.each(['../../admin?x=1', 'd laptop', 'd/1', '', 'x'.repeat(129)])(
    'useDeliverable(%j) leaves the host idle: a value that is not a Deliverable id is never asked for',
    (id) => {
      renderHook(() => useDeliverable(id, true))
      expect(lastCall('useDeliverable')).toEqual([null, true])
    },
  )

  it("useFileDownload and useAuthenticatedBlobUrl are the host's, with the theme's arguments", () => {
    const { result: downloader } = renderHook(() => useFileDownload('Try again'))
    expect(lastCall('useApiFileDownload')).toEqual(['Try again'])
    expect(downloader.current).toBe(host.download)
    const { result: blob } = renderHook(() => useAuthenticatedBlobUrl('/api/deliverables/d-1/file', 'image/png'))
    expect(lastCall('useAuthenticatedBlobUrl')).toEqual(['/api/deliverables/d-1/file', 'image/png'])
    expect(blob.current).toBe(host.blob)
  })
})

describe('PRD-261 A-12a — the Deliverables constants', () => {
  it("DELIVERABLE_FEED_FILTERS is a frozen copy of the host's feed filters", () => {
    expect(DELIVERABLE_FEED_FILTERS).toEqual(FEED_DEFAULT_FILTERS)
    expect(DELIVERABLE_FEED_FILTERS).not.toBe(FEED_DEFAULT_FILTERS)
    expect(Object.isFrozen(DELIVERABLE_FEED_FILTERS)).toBe(true)
    expect(Object.isFrozen(DELIVERABLE_FEED_FILTERS.source_type_exclude)).toBe(true)
    expect(DELIVERABLE_FEED_FILTERS.source_type_exclude).not.toBe(FEED_DEFAULT_FILTERS.source_type_exclude)
  })

  it("DELIVERABLE_TAB_LABELS is a frozen copy of the host's tab labels", () => {
    expect(DELIVERABLE_TAB_LABELS).toEqual(TAB_LABELS)
    expect(DELIVERABLE_TAB_LABELS).not.toBe(TAB_LABELS)
    expect(Object.isFrozen(DELIVERABLE_TAB_LABELS)).toBe(true)
  })

  it('visibleDeliverableTabs shows Socials only while the platform offers it, as the host does', () => {
    expect(visibleDeliverableTabs(true)).toEqual(hostVisibleTabs(true))
    expect(visibleDeliverableTabs(false)).toEqual(hostVisibleTabs(false))
    expect(visibleDeliverableTabs(false)).not.toContain('socials')
  })

  it("deliverableLabel is the host's name for a kind it knows, and the kind itself for one it does not", () => {
    expect(deliverableLabel('report')).toBe(hostLabel('report'))
    expect(deliverableLabel('blog_post')).toBe(hostLabel('blog_post'))
    expect(deliverableLabel('podcast')).toBe('podcast')
  })

  it("the URL and name helpers are the host's own", () => {
    expect(deliverableFileUrl).toBe(download.deliverableFileUrl)
    expect(deliverableDownloadUrl).toBe(download.deliverableDownloadUrl)
    expect(deliverableFilename).toBe(download.deliverableFilename)
    expect(fileDownloadUrl).toBe(download.fileDownloadUrl)
  })
})
