/**
 * PRD-261 A-01 — the global handoff. The host's React, ReactDOM, jsx runtime and SDK go on one
 * frozen, non-writable, non-configurable global, once. A later call returns the same object, and
 * refuses a global that carries another React. The tests run in order: the first one installs a
 * configurable fake and deletes it, because the real global cannot be removed once it is set.
 */
import * as React from 'react'
import { describe, expect, it, vi } from 'vitest'

import sharedModules from '../shared-modules.json'
import { exposeHostModules } from '../host-modules'

const GLOBAL = sharedModules.hostGlobal
const hostWindow = window as unknown as Record<string, unknown>

describe('PRD-261 A-01 — exposeHostModules', () => {
  it('refuses a global whose React is another instance', () => {
    const otherReact = { useState: () => undefined }
    Object.defineProperty(window, GLOBAL, {
      value: { sdkVersion: sharedModules.sdkVersion, modules: { react: otherReact } },
      writable: false,
      configurable: true,
    })
    try {
      expect(() => exposeHostModules()).toThrow('Automatos theme host: a second React is on this page.')
    } finally {
      delete hostWindow[GLOBAL]
    }
    expect(GLOBAL in window).toBe(false)
  })

  it('installs one frozen, non-writable, non-configurable global keyed by every shared specifier', () => {
    const host = exposeHostModules()
    const descriptor = Object.getOwnPropertyDescriptor(window, GLOBAL)
    expect(descriptor?.value).toBe(host)
    expect(descriptor?.writable).toBe(false)
    expect(descriptor?.configurable).toBe(false)
    expect(descriptor?.enumerable).toBe(false)
    expect(Object.isFrozen(host)).toBe(true)
    expect(Object.isFrozen(host.modules)).toBe(true)
    expect(host.sdkVersion).toBe(sharedModules.sdkVersion)
    expect(Object.keys(host.modules).sort()).toEqual(sharedModules.modules.map((entry) => entry.specifier).sort())
    expect((host.modules.react as typeof React).useState).toBe(React.useState)
  })

  it('returns the same object on a second call', () => {
    expect(exposeHostModules()).toBe(exposeHostModules())
  })

  it('returns the same object from a fresh copy of the module: a new namespace, the same React', async () => {
    const first = exposeHostModules()
    vi.resetModules()
    const fresh = await import('../host-modules')
    expect(() => fresh.exposeHostModules()).not.toThrow()
    expect(fresh.exposeHostModules()).toBe(first)
  })
})
