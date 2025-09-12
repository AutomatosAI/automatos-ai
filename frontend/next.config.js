/** @type {import('next').NextConfig} */
const nextConfig = {
  typescript: { 
    ignoreBuildErrors: true 
  },
  logging: {
    fetches: {
      fullUrl: true
    }
  },
  experimental: {
    logging: {
      level: 'verbose'
    }
  }
}

module.exports = nextConfig
