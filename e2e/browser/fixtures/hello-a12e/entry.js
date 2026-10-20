// PRD-261 A-12e: the missions-data hello theme. Besides a frame that only wraps the page, it
// registers its own page on both missions routes (Gerard's Q3 "missions", D0), and each reads its
// data through the SDK's pages/assignments area alone:
// - /assignments (the Missions tab): useMissions() and useRecommendedAssignments(), each Mission
//   linked with missionHref and HostLink;
// - /missions/[id]: the id from useHostPathname(), the Mission from useMission(), and a Pause
//   button that runs usePauseMission().
// It imports nothing of the host but the shared modules. It reaches the browser only through the
// lane's route interception (e2e/browser/specs/pages-assignments.spec.ts, through
// support/hello-theme.ts), which also answers the Mission endpoints from responses.json.
import { jsx, jsxs } from 'react/jsx-runtime'
import {
  defineTheme,
  HostLink,
  missionHref,
  useHostPathname,
  useMission,
  useMissions,
  usePauseMission,
  useRecommendedAssignments,
} from '@automatos/theme-sdk'

const LIST_LIMIT = 10
const RECOMMENDED_LIMIT = 3
/** '/missions/<id>' split on '/': ['', 'missions', '<id>']. */
const MISSION_PATH_PARTS = 3
const MISSION_ID_PART = 2

/** @param {{ children?: import('react').ReactNode }} props */
function MissionsFrame({ children }) {
  return jsx('div', { 'data-testid': 'hello-a12e-frame', className: 'hello-a12e-frame', children })
}

/** @param {{ mission: { id: string, goal: string, state: string } }} props */
function MissionRow({ mission }) {
  return jsx('li', {
    'data-testid': 'hello-a12e-mission',
    'data-mission-id': mission.id,
    'data-state': mission.state,
    children: jsx(HostLink, { href: missionHref(mission.id), children: mission.goal }),
  })
}

function HelloMissionsPage() {
  const missions = useMissions({ limit: LIST_LIMIT })
  const recommended = useRecommendedAssignments(RECOMMENDED_LIMIT)
  const rows = missions.data ? missions.data.missions : []
  const items = recommended.data ? recommended.data.items : []
  return jsxs('section', {
    'data-testid': 'hello-a12e-missions-page',
    className: 'hello-a12e-page',
    'aria-label': 'Hello missions',
    children: [
      jsx('h2', { children: 'Missions, read through the SDK' }),
      jsx('p', { 'data-testid': 'hello-a12e-total', children: missions.data ? String(missions.data.total) : '' }),
      jsx('ul', { 'data-testid': 'hello-a12e-missions', children: rows.map((mission) => jsx(MissionRow, { mission }, mission.id)) }),
      jsx('ul', {
        'data-testid': 'hello-a12e-recommended',
        children: items.map((item) => jsx('li', { 'data-testid': 'hello-a12e-recommended-item', children: item.name }, String(item.id))),
      }),
    ],
  })
}

/** @param {string} pathname */
function missionIdOf(pathname) {
  const parts = pathname.split('/')
  return parts.length === MISSION_PATH_PARTS ? parts[MISSION_ID_PART] : null
}

function HelloMissionPage() {
  const missionId = missionIdOf(useHostPathname())
  const mission = useMission(missionId)
  const pause = usePauseMission()
  const detail = mission.data
  return jsxs('section', {
    'data-testid': 'hello-a12e-mission-page',
    className: 'hello-a12e-page',
    'aria-label': 'Hello mission',
    children: [
      jsx('h2', { 'data-testid': 'hello-a12e-goal', children: detail ? detail.goal : '' }),
      jsx('p', { 'data-testid': 'hello-a12e-state', children: detail ? detail.state : '' }),
      jsx('p', { 'data-testid': 'hello-a12e-tasks', children: detail ? String(detail.tasks.length) : '' }),
      jsx('button', {
        type: 'button',
        'data-testid': 'hello-a12e-pause',
        disabled: detail === null || missionId === null || pause.isRunning,
        onClick: () => {
          if (missionId !== null) pause.run(missionId)
        },
        children: 'Pause',
      }),
      jsx('p', { 'data-testid': 'hello-a12e-paused-state', children: pause.result ? pause.result.state : '' }),
    ],
  })
}

export default defineTheme({
  register(host) {
    host.registerSlot('frame@1', MissionsFrame)
    host.registerPage('/assignments', HelloMissionsPage)
    host.registerPage('/missions/[id]', HelloMissionPage)
  },
  unregister() {},
})
