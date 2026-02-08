/** @type {import('next').NextConfig} */
const path = require('path')

const nextConfig = {
  reactStrictMode: false,
  typescript: {
    ignoreBuildErrors: true
  },
  turbopack: {
    root: __dirname,
  },
  // Rewrites disabled - using absolute URLs from client instead
  // Next.js rewrites require env vars at build time, which is problematic on Railway
  // The API client will use NEXT_PUBLIC_API_URL for direct calls to backend
}

module.exports = nextConfig
