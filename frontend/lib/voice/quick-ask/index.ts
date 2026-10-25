/** Theme voice, web quick-ask: the engine the host's voice provider runs for hold to talk in the browser. */
export { createQuickAskEngine } from './create-engine'
export type { QuickAskDeps } from './deps'
export type { RecognitionEvents, RecognitionFactory, RecognitionFailure, RecognitionPort } from './recognition'
export type { SynthesisPort, UtteranceEvents } from './speaker'
export type { QuickAskClock } from './clock'
