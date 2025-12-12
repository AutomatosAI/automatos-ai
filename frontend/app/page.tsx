import { redirect } from 'next/navigation'

 export default function Home() {
  // New home is Chat
  // Keep this as a server redirect to avoid rendering dashboard on "/"
  // and to make "/dashboard" the dedicated dashboard route.
  redirect('/chat')
 }
