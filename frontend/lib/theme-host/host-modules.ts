/**
 * PRD-261 A-01 — the global handoff: the host's own React, ReactDOM, jsx runtime and theme SDK
 * on one frozen global, which the shims under public/theme-sdk/v1 re-export to a theme.
 *
 * Next's React lives inside webpack chunks, not as an ES module a browser can import. In the
 * App Router's client bundle these imports resolve to next/dist/compiled/react…, the same
 * modules the page hydrates with, so a theme that imports 'react' through the import map gets
 * the instance the page renders with. Only the loader chunk imports this file, so it is not in
 * the page's first bundle and safe mode never evaluates it.
 */
import * as React from 'react'
import * as ReactDOM from 'react-dom'
import * as ReactDOMClient from 'react-dom/client'
import * as JsxRuntime from 'react/jsx-runtime'
import * as ThemeSdk from './sdk'
import sharedModules from './shared-modules.json'
import type { ThemeHostGlobal } from './types'

const SECOND_REACT_MESSAGE = 'Automatos theme host: a second React is on this page.'

/** The host's instance of each specifier in shared-modules.json. */
const HOST_INSTANCES: Readonly<Record<string, unknown>> = {
  react: React,
  'react/jsx-runtime': JsxRuntime,
  'react-dom': ReactDOM,
  'react-dom/client': ReactDOMClient,
  '@automatos/theme-sdk': ThemeSdk,
}

function hostInstance(specifier: string): unknown {
  if (!Object.prototype.hasOwnProperty.call(HOST_INSTANCES, specifier)) {
    throw new Error(`Automatos theme host: no host instance for "${specifier}".`)
  }
  return HOST_INSTANCES[specifier]
}

function existingHost(): ThemeHostGlobal | undefined {
  return (window as unknown as Record<string, ThemeHostGlobal | undefined>)[sharedModules.hostGlobal]
}

/**
 * Installs the global once and returns it; a later call returns the same object. The check on
 * a later call compares the React INSTANCE (its useState), not the namespace object: webpack
 * builds a new namespace object over the same module.exports whenever this file is evaluated
 * again (a hot reload, a test's module reset), and that is still the same React.
 */
export function exposeHostModules(): ThemeHostGlobal {
  const existing = existingHost()
  if (existing) {
    if ((existing.modules.react as typeof React | undefined)?.useState !== React.useState) {
      throw new Error(SECOND_REACT_MESSAGE)
    }
    return existing
  }
  const modules = Object.fromEntries(sharedModules.modules.map((entry) => [entry.specifier, hostInstance(entry.specifier)]))
  const value: ThemeHostGlobal = Object.freeze({ sdkVersion: sharedModules.sdkVersion, modules: Object.freeze(modules) })
  Object.defineProperty(window, sharedModules.hostGlobal, { value, writable: false, configurable: false, enumerable: false })
  return value
}
