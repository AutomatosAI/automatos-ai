/** @type {import('next').NextConfig} */
const nextConfig = {
  typescript: { 
    ignoreBuildErrors: true 
  },
  // Commented out proxy - we'll handle API calls with mock fallback
  // async rewrites() {
  //   return [
  //     {
  //       source: '/api/:path*',
  //       destination: 'http://localhost:8000/api/:path*',
  //     },
  //   ]
  // }
}

module.exports = nextConfig
