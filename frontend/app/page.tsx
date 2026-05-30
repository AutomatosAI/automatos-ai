import { redirect } from 'next/navigation'

 export default function Home() {
  // New home is Chat
  redirect('/chat')
 }
