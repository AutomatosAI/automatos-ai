'use client'

import { motion } from 'framer-motion'
import { Menu, Bell, User } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { ThemeToggle } from '@/components/ui/theme-toggle'

interface HeaderProps {
  onMenuClick: () => void
}

export function Header({ onMenuClick }: HeaderProps) {
  return (
    <header className="sticky top-0 z-30 glass-card border-b border-border/50 px-6 py-4">
      <div className="flex items-center justify-between">
        {/* Left side */}
        <div className="flex items-center space-x-4">
          <Button
            variant="ghost"
            size="icon"
            onClick={onMenuClick}
            className="lg:hidden text-orange-400 hover:text-orange-300 hover:bg-orange-500/5"
          >
            <Menu className="w-5 h-5" />
          </Button>

          {/* Brand (replaces non-functional search) */}
          <div className="flex items-center">
            <img
              src="/brand/automatos-logo.png"
              alt="Automatos A.I."
              className="h-8 w-auto opacity-95"
              draggable={false}
            />
          </div>
        </div>

        {/* Right side */}
        <div className="flex items-center space-x-4">
          {/* Theme Toggle */}
          <ThemeToggle />

          {/* Notifications */}
          <Button
            variant="ghost"
            size="icon"
            className="relative text-orange-400 hover:text-orange-300 hover:bg-orange-500/5"
          >
            <Bell className="w-5 h-5" />
            <span className="absolute -top-1 -right-1 w-3 h-3 bg-orange-500 rounded-full text-[10px] flex items-center justify-center text-black">
              3
            </span>
          </Button>

          {/* User Menu */}
          <Button
            variant="ghost"
            size="icon"
            className="rounded-full text-orange-400 hover:text-orange-300 hover:bg-orange-500/5"
          >
            <User className="w-5 h-5" />
          </Button>
        </div>
      </div>
    </header>
  )
}
