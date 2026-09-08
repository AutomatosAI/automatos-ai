import { describe, expect, it } from 'vitest'
import { runtimeCanvasData, runtimeCanvasRoot, runtimeCanvasTitle, sessionCanvasHref } from '@/lib/chat/runtime-canvas'

describe('Runtime Canvas (PRD-239 S7 v2)', () => {
  it('opens on the browsable root, else the workspace root', () => {
    expect(runtimeCanvasRoot('sessions/93')).toBe('sessions/93')
    expect(runtimeCanvasRoot(null)).toBe('.')
    expect(runtimeCanvasRoot('  ')).toBe('.')
  })

  it('the board deep-links into the same view', () => {
    expect(sessionCanvasHref(93, 'projects/automatos-ai')).toBe('/chat?repo=projects%2Fautomatos-ai&ticket=93&runtime=1')
    expect(sessionCanvasHref('7', null)).toBe('/chat?repo=.&ticket=7&runtime=1')
  })

  it('the widget opens fullscreen, rooted at the session folder, marked runtime', () => {
    expect(runtimeCanvasData('ws-1', { task_id: 93, explorer_root: 'sessions/93', agent_name: 'Bob' })).toEqual({
      workspaceId: 'ws-1', rootPath: 'sessions/93', taskId: '93', runtime: true, openFullscreen: true,
    })
    expect(runtimeCanvasTitle({ task_id: 93, agent_name: 'Bob' })).toBe('Bob · session')
    expect(runtimeCanvasTitle({ task_id: 93 })).toBe('Session · ticket #93')
  })
})
