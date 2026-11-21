/**
 * PRD-277 US-704 (D-12) — which of Automatos's hosted services this install calls.
 *
 * - the local build calls every one, the hosted build none, and neither asks;
 * - the enterprise build knows none of them and asks (`editionHostedServices` is null);
 * - every build but the hosted one may hold Cloud items (`holdsCloudItems`), whatever the settings say;
 * - an answer turns a service on only with `true`: anything else, a missing field or a malformed answer is off;
 * - `readHostedServices` reads `GET /api/hosted-services` through the client it is handed.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'

type HostedServicesModule = typeof import('../hosted-services')

const ALL_ON = { themeRevocations: true, cloudSync: true, cloudItems: true }
const ALL_OFF = { themeRevocations: false, cloudSync: false, cloudItems: false }

async function load(edition: 'local' | 'saas' | 'enterprise'): Promise<HostedServicesModule> {
  vi.resetModules()
  vi.doMock('@/lib/edition-name', () => ({ isEnterprise: edition === 'enterprise', resolveEdition: () => edition }))
  return import('../hosted-services')
}

afterEach(() => {
  vi.doUnmock('@/lib/edition-name')
})

describe('what the build knows', () => {
  it('the local build calls every service', async () => {
    expect((await load('local')).editionHostedServices()).toEqual(ALL_ON)
  })

  it('the hosted build calls none: it is Automatos Cloud', async () => {
    expect((await load('saas')).editionHostedServices()).toEqual(ALL_OFF)
  })

  it('the enterprise build knows none and asks', async () => {
    expect((await load('enterprise')).editionHostedServices()).toBeNull()
  })
})

describe('what the install may hold', () => {
  // A setting stops requests, never what is held: Cloud items are shown and reviewed everywhere but the host.
  it.each([
    ['local', true],
    ['enterprise', true],
    ['saas', false],
  ] as const)('the %s build holds Cloud items: %s', async (edition, holds) => {
    expect((await load(edition)).holdsCloudItems()).toBe(holds)
  })
})

describe('the answer of GET /api/hosted-services', () => {
  it('turns a service on only with true', async () => {
    const { hostedServicesOf } = await load('enterprise')
    expect(hostedServicesOf(ALL_ON)).toEqual(ALL_ON)
    expect(hostedServicesOf({ themeRevocations: true, cloudSync: 'true', cloudItems: 1 })).toEqual({
      themeRevocations: true, cloudSync: false, cloudItems: false,
    })
  })

  it.each([null, undefined, 'on', 42, [], {}])('reads %j as every service off', async (raw) => {
    const { hostedServicesOf } = await load('enterprise')
    expect(hostedServicesOf(raw)).toEqual(ALL_OFF)
  })

  it('is read through the client it is handed', async () => {
    const { HOSTED_SERVICES_ENDPOINT, readHostedServices } = await load('enterprise')
    const get = vi.fn(async () => ({ themeRevocations: true, cloudSync: false, cloudItems: false }))

    await expect(readHostedServices({ get } as never)).resolves.toEqual({
      themeRevocations: true, cloudSync: false, cloudItems: false,
    })
    expect(HOSTED_SERVICES_ENDPOINT).toBe('/api/hosted-services')
    expect(get).toHaveBeenCalledWith(HOSTED_SERVICES_ENDPOINT)
  })
})
