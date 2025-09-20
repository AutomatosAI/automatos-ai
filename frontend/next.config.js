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
  },
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://localhost:8000/api/:path*',
      },
    ]
  }
}

module.exports = nextConfig
