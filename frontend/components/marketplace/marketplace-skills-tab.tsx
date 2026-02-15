'use client'

import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { Search, Download, Trash2, Shield, ShieldAlert, CheckCircle2, Loader2 } from 'lucide-react'

interface Skill {
  name: string
  description: string
  author: string
  category: string
  installs: number
  installed?: boolean
  source?: string
}

export function MarketplaceSkillsTab({ searchQuery }: { searchQuery: string }) {
  const [skills, setSkills] = useState<Skill[]>([])
  const [installed, setInstalled] = useState<Skill[]>([])
  const [loading, setLoading] = useState(false)
  const [installing, setInstalling] = useState<string | null>(null)
  const [localSearch, setLocalSearch] = useState(searchQuery || '')

  useEffect(() => {
    // Load installed skills
    fetch('/api/skills/community/installed')
      .then(r => r.json())
      .then(data => setInstalled(data.skills || []))
      .catch(() => {})
  }, [])

  const searchSkills = async () => {
    if (!localSearch.trim()) return
    setLoading(true)
    try {
      const res = await fetch(`/api/skills/community/search?q=${encodeURIComponent(localSearch)}`)
      const data = await res.json()
      setSkills(data.results || [])
    } finally {
      setLoading(false)
    }
  }

  const installSkill = async (skillName: string) => {
    setInstalling(skillName)
    try {
      const res = await fetch('/api/skills/community/install', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ skill_name: skillName })
      })
      const data = await res.json()
      if (data.status === 'installed' || data.status === 'success') {
        setInstalled(prev => [...prev, { name: skillName, description: '', author: '', category: '', installs: 0, source: 'workspace' }])
      } else if (data.status === 'blocked') {
        alert(`Skill blocked: ${data.reason || 'Security risk too high'}`)
      } else if (data.status === 'review_required') {
        if (confirm(`Security review required (risk score: ${data.risk_score}). Install anyway?`)) {
          // Re-install with force flag
          await fetch('/api/skills/community/install', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ skill_name: skillName, force: true })
          })
          setInstalled(prev => [...prev, { name: skillName, description: '', author: '', category: '', installs: 0, source: 'workspace' }])
        }
      }
    } finally {
      setInstalling(null)
    }
  }

  const uninstallSkill = async (skillName: string) => {
    if (!confirm(`Uninstall ${skillName}?`)) return
    await fetch(`/api/skills/community/${skillName}`, { method: 'DELETE' })
    setInstalled(prev => prev.filter(s => s.name !== skillName))
  }

  const installedNames = new Set(installed.map(s => s.name))

  return (
    <div className="space-y-6">
      {/* Search */}
      <div className="flex gap-2">
        <Input
          placeholder="Search community skills..."
          value={localSearch}
          onChange={e => setLocalSearch(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && searchSkills()}
        />
        <Button onClick={searchSkills} disabled={loading}>
          {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
        </Button>
      </div>

      {/* Search Results */}
      {skills.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {skills.map(skill => (
            <Card key={skill.name} className="border-border/40 bg-card/50">
              <CardHeader className="pb-2">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-sm">{skill.name}</CardTitle>
                  <Badge variant="outline">{skill.category}</Badge>
                </div>
                <CardDescription className="text-xs">{skill.description}</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-muted-foreground">by {skill.author} · {skill.installs} installs</span>
                  {installedNames.has(skill.name) ? (
                    <Badge variant="secondary"><CheckCircle2 className="h-3 w-3 mr-1" /> Installed</Badge>
                  ) : (
                    <Button size="sm" variant="outline" onClick={() => installSkill(skill.name)} disabled={installing === skill.name}>
                      {installing === skill.name ? <Loader2 className="h-3 w-3 animate-spin mr-1" /> : <Download className="h-3 w-3 mr-1" />}
                      Install
                    </Button>
                  )}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* Installed Skills */}
      <div>
        <h3 className="text-lg font-semibold mb-3">Installed Skills</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {installed.map(skill => (
            <Card key={skill.name} className="border-border/40 bg-card/50">
              <CardContent className="pt-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-medium text-sm">{skill.name}</p>
                    <Badge variant="outline" className="text-xs mt-1">
                      {skill.source === 'built-in' ? <Shield className="h-3 w-3 mr-1" /> : null}
                      {skill.source || 'workspace'}
                    </Badge>
                  </div>
                  {skill.source !== 'built-in' && (
                    <Button size="sm" variant="ghost" onClick={() => uninstallSkill(skill.name)}>
                      <Trash2 className="h-3 w-3" />
                    </Button>
                  )}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </div>
  )
}
