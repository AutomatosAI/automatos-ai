
import type { Metadata } from 'next'
import { Providers } from '../components/providers'
import './globals.css'

export const metadata: Metadata = {
  title: 'Automatos AI Platform',
  description: 'Enterprise AI automation and agent management platform',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>
        <Providers>
          {children}
        </Providers>
      </body>
    </html>
  )
}
