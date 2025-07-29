
import './globals.css'
import { Inter } from 'next/font/google'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Toaster } from 'react-hot-toast'
import { Providers } from '@/components/providers'

const inter = Inter({ subsets: ['latin'] })

export const metadata = {
  title: 'Automotas AI - Multi-Agent Orchestration Platform',
  description: 'Advanced multi-agent orchestration platform with context engineering, workflow management, and intelligent automation for enterprise AI systems.',
  keywords: 'AI agents, orchestration, context engineering, workflow automation, multi-agent systems, enterprise AI',
  authors: [{ name: 'Automotas AI Team' }],
  creator: 'Automotas AI',
  publisher: 'Automotas AI',
  formatDetection: {
    email: false,
    address: false,
    telephone: false,
  },
  openGraph: {
    title: 'Automotas AI - Multi-Agent Orchestration Platform',
    description: 'Advanced multi-agent orchestration platform with context engineering and workflow management.',
    url: 'https://automotas.ai',
    siteName: 'Automotas AI',
    locale: 'en_US',
    type: 'website',
  },
}

export const viewport = {
  width: 'device-width',
  initialScale: 1,
  maximumScale: 1,
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.className} gradient-bg min-h-screen antialiased`}>
        <Providers>
          {children}
          <Toaster 
            position="top-right"
            toastOptions={{
              duration: 4000,
              style: {
                background: 'hsl(var(--card))',
                color: 'hsl(var(--card-foreground))',
                border: '1px solid hsl(var(--border))',
              },
            }}
          />
        </Providers>
      </body>
    </html>
  )
}
