import { defineConfig, devices } from '@playwright/test'

// PRD-261 A-00: the browser lane. CI builds the frontend, serves its standalone output on
// http://localhost:3000 (the local backend's CORS default) and runs these specs against it.

const DEFAULT_APP_URL = 'http://localhost:3000'
const TEST_TIMEOUT_MS = 60_000
const EXPECT_TIMEOUT_MS = 15_000

// 1440 px keeps the Studio desktop chrome (MainLayout switches below 1024 px). It goes AFTER
// the device spread in each project: every devices[...] entry carries its own 1280x720, and
// a project's `use` is merged over the top-level `use`.
const DESKTOP_VIEWPORT = { width: 1440, height: 900 }

export default defineConfig({
  testDir: 'specs',
  fullyParallel: false,
  workers: 2,
  // A flake must show as red, not pass on a second try.
  retries: 0,
  timeout: TEST_TIMEOUT_MS,
  expect: { timeout: EXPECT_TIMEOUT_MS },
  reporter: [['list'], ['github'], ['html', { open: 'never' }]],
  use: {
    baseURL: process.env.E2E_APP_URL ?? DEFAULT_APP_URL,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'], viewport: DESKTOP_VIEWPORT },
    },
    {
      name: 'webkit',
      use: { ...devices['Desktop Safari'], viewport: DESKTOP_VIEWPORT },
    },
    // PRD-261 A-01: Firefox runs the @record specs only (the React version and import-map
    // integrity support); the lane itself is Chromium and WebKit.
    {
      name: 'firefox',
      use: { ...devices['Desktop Firefox'], viewport: DESKTOP_VIEWPORT },
      grep: /@record/,
    },
  ],
})
