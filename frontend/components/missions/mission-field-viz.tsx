'use client'

import { useRef, useEffect, useMemo, useCallback } from 'react'
import { cn } from '@/lib/utils'
import type { FieldPattern } from '@/hooks/use-missions-api'

interface MissionFieldVizProps {
  patterns: FieldPattern[]
  className?: string
}

// Agent color palette - vibrant for dark backgrounds
const AGENT_PALETTE: Record<number, { r: number; g: number; b: number }> = {}
const PALETTE = [
  { r: 59, g: 130, b: 246 },   // blue
  { r: 16, g: 185, b: 129 },   // emerald
  { r: 168, g: 85, b: 247 },   // purple
  { r: 245, g: 158, b: 11 },   // amber
  { r: 244, g: 63, b: 94 },    // rose
  { r: 6, g: 182, b: 212 },    // cyan
  { r: 249, g: 115, b: 22 },   // orange
  { r: 132, g: 204, b: 22 },   // lime
]

function getAgentPalette(agentId: number, index: number) {
  if (!AGENT_PALETTE[agentId]) {
    AGENT_PALETTE[agentId] = PALETTE[index % PALETTE.length]
  }
  return AGENT_PALETTE[agentId]
}

interface VizNode {
  id: string
  x: number
  y: number
  vx: number
  vy: number
  radius: number
  color: { r: number; g: number; b: number }
  strength: number
  accessCount: number
  agentId: number
  key: string
  isAgent: boolean
  pulsePhase: number
  targetX: number
  targetY: number
}

interface Particle {
  x: number
  y: number
  vx: number
  vy: number
  life: number
  maxLife: number
  color: { r: number; g: number; b: number }
  size: number
}

export function MissionFieldViz({ patterns, className }: MissionFieldVizProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animFrameRef = useRef<number>(0)
  const nodesRef = useRef<VizNode[]>([])
  const particlesRef = useRef<Particle[]>([])
  const timeRef = useRef(0)
  const lastPatternCountRef = useRef(0)

  // Build nodes from patterns
  const { nodes: initialNodes, agentNodes } = useMemo(() => {
    const uniqueAgents = [...new Set(patterns.map(p => p.agent_id))]
    const agentIndexMap = new Map(uniqueAgents.map((id, i) => [id, i]))

    // Create agent hub nodes
    const agentNodes: VizNode[] = uniqueAgents.map((agentId, i) => {
      const angle = (i / uniqueAgents.length) * Math.PI * 2 - Math.PI / 2
      const hubRadius = Math.min(200, 80 + uniqueAgents.length * 20)
      return {
        id: `agent_${agentId}`,
        x: 0.5 + Math.cos(angle) * 0.25,
        y: 0.5 + Math.sin(angle) * 0.25,
        vx: 0,
        vy: 0,
        radius: 18,
        color: getAgentPalette(agentId, i),
        strength: 1,
        accessCount: 0,
        agentId,
        key: agentId === 0 ? 'System' : `Agent ${agentId}`,
        isAgent: true,
        pulsePhase: Math.random() * Math.PI * 2,
        targetX: 0.5 + Math.cos(angle) * 0.25,
        targetY: 0.5 + Math.sin(angle) * 0.25,
      }
    })

    // Create pattern nodes orbiting their agent
    const patternNodes: VizNode[] = patterns.map((p, i) => {
      const agentIdx = agentIndexMap.get(p.agent_id) ?? 0
      const agentNode = agentNodes[agentIdx]
      const patternsForAgent = patterns.filter(pp => pp.agent_id === p.agent_id)
      const indexInAgent = patternsForAgent.indexOf(p)
      const angleOffset = (indexInAgent / patternsForAgent.length) * Math.PI * 2
      const orbitRadius = 0.08 + Math.random() * 0.12

      return {
        id: p.id,
        x: agentNode.x + Math.cos(angleOffset) * orbitRadius,
        y: agentNode.y + Math.sin(angleOffset) * orbitRadius,
        vx: (Math.random() - 0.5) * 0.0005,
        vy: (Math.random() - 0.5) * 0.0005,
        radius: 4 + p.decayed_strength * 10,
        color: getAgentPalette(p.agent_id, agentIdx),
        strength: p.decayed_strength,
        accessCount: p.access_count,
        agentId: p.agent_id,
        key: p.key,
        isAgent: false,
        pulsePhase: Math.random() * Math.PI * 2,
        targetX: agentNode.x + Math.cos(angleOffset) * orbitRadius,
        targetY: agentNode.y + Math.sin(angleOffset) * orbitRadius,
      }
    })

    return { nodes: [...agentNodes, ...patternNodes], agentNodes }
  }, [patterns])

  // Spawn particles when new patterns arrive
  const spawnParticles = useCallback((node: VizNode, count: number) => {
    for (let i = 0; i < count; i++) {
      const angle = Math.random() * Math.PI * 2
      const speed = 0.001 + Math.random() * 0.002
      particlesRef.current.push({
        x: node.x,
        y: node.y,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed,
        life: 1,
        maxLife: 60 + Math.random() * 90,
        color: node.color,
        size: 1 + Math.random() * 2,
      })
    }
  }, [])

  // Update nodes when patterns change
  useEffect(() => {
    const existingIds = new Set(nodesRef.current.map(n => n.id))
    const newNodes = initialNodes.filter(n => !existingIds.has(n.id))

    // Update existing nodes' properties
    for (const node of nodesRef.current) {
      const updated = initialNodes.find(n => n.id === node.id)
      if (updated) {
        node.strength = updated.strength
        node.radius = updated.radius
        node.accessCount = updated.accessCount
      }
    }

    // Add new nodes with particle burst
    for (const node of newNodes) {
      nodesRef.current.push(node)
      if (!node.isAgent) {
        spawnParticles(node, 15)
      }
    }

    // Spawn ambient particles from agents
    if (patterns.length > lastPatternCountRef.current) {
      for (const agentNode of nodesRef.current.filter(n => n.isAgent)) {
        spawnParticles(agentNode, 8)
      }
    }
    lastPatternCountRef.current = patterns.length
  }, [initialNodes, patterns.length, spawnParticles])

  // Animation loop
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const resize = () => {
      const dpr = window.devicePixelRatio || 1
      const rect = canvas.getBoundingClientRect()
      canvas.width = rect.width * dpr
      canvas.height = rect.height * dpr
      ctx.scale(dpr, dpr)
    }
    resize()
    window.addEventListener('resize', resize)

    const animate = () => {
      const rect = canvas.getBoundingClientRect()
      const w = rect.width
      const h = rect.height
      timeRef.current += 0.016

      // Clear with fade trail
      ctx.fillStyle = 'rgba(10, 10, 14, 0.15)'
      ctx.fillRect(0, 0, w, h)

      const nodes = nodesRef.current
      const particles = particlesRef.current

      // Simple force simulation
      for (const node of nodes) {
        if (node.isAgent) continue

        // Drift toward target
        node.vx += (node.targetX - node.x) * 0.0003
        node.vy += (node.targetY - node.y) * 0.0003

        // Gentle orbital motion
        const agentNode = nodes.find(n => n.isAgent && n.agentId === node.agentId)
        if (agentNode) {
          const dx = node.x - agentNode.x
          const dy = node.y - agentNode.y
          // Perpendicular force for orbiting
          node.vx += -dy * 0.00003
          node.vy += dx * 0.00003
        }

        // Repulsion between pattern nodes
        for (const other of nodes) {
          if (other.id === node.id || other.isAgent) continue
          const dx = node.x - other.x
          const dy = node.y - other.y
          const dist = Math.sqrt(dx * dx + dy * dy)
          if (dist < 0.08 && dist > 0.001) {
            const force = 0.00001 / (dist * dist)
            node.vx += dx * force
            node.vy += dy * force
          }
        }

        // Damping
        node.vx *= 0.98
        node.vy *= 0.98

        node.x += node.vx
        node.y += node.vy

        // Boundary
        node.x = Math.max(0.05, Math.min(0.95, node.x))
        node.y = Math.max(0.05, Math.min(0.95, node.y))
      }

      // Draw connections from patterns to their agent
      for (const node of nodes) {
        if (node.isAgent) continue
        const agentNode = nodes.find(n => n.isAgent && n.agentId === node.agentId)
        if (!agentNode) continue

        const alpha = node.strength * 0.3
        ctx.beginPath()
        ctx.strokeStyle = `rgba(${node.color.r}, ${node.color.g}, ${node.color.b}, ${alpha})`
        ctx.lineWidth = 0.5 + node.strength
        ctx.moveTo(agentNode.x * w, agentNode.y * h)
        ctx.lineTo(node.x * w, node.y * h)
        ctx.stroke()
      }

      // Draw cross-agent connections (patterns that were co-accessed)
      for (let i = 0; i < nodes.length; i++) {
        const a = nodes[i]
        if (a.isAgent || a.accessCount === 0) continue
        for (let j = i + 1; j < nodes.length; j++) {
          const b = nodes[j]
          if (b.isAgent || b.accessCount === 0) continue
          if (a.agentId === b.agentId) continue

          const dx = a.x - b.x
          const dy = a.y - b.y
          const dist = Math.sqrt(dx * dx + dy * dy)
          if (dist < 0.3) {
            const alpha = Math.min(a.accessCount, b.accessCount) * 0.05 * (1 - dist / 0.3)
            ctx.beginPath()
            ctx.strokeStyle = `rgba(255, 255, 255, ${alpha})`
            ctx.lineWidth = 0.5
            ctx.setLineDash([2, 4])
            ctx.moveTo(a.x * w, a.y * h)
            ctx.lineTo(b.x * w, b.y * h)
            ctx.stroke()
            ctx.setLineDash([])
          }
        }
      }

      // Draw pattern nodes
      for (const node of nodes) {
        if (node.isAgent) continue

        const pulse = Math.sin(timeRef.current * 2 + node.pulsePhase) * 0.3 + 0.7
        const alpha = 0.3 + node.strength * 0.7
        const r = node.radius * pulse

        // Outer glow
        const gradient = ctx.createRadialGradient(
          node.x * w, node.y * h, 0,
          node.x * w, node.y * h, r * 3,
        )
        gradient.addColorStop(0, `rgba(${node.color.r}, ${node.color.g}, ${node.color.b}, ${alpha * 0.4})`)
        gradient.addColorStop(0.5, `rgba(${node.color.r}, ${node.color.g}, ${node.color.b}, ${alpha * 0.1})`)
        gradient.addColorStop(1, `rgba(${node.color.r}, ${node.color.g}, ${node.color.b}, 0)`)
        ctx.fillStyle = gradient
        ctx.beginPath()
        ctx.arc(node.x * w, node.y * h, r * 3, 0, Math.PI * 2)
        ctx.fill()

        // Core
        ctx.fillStyle = `rgba(${node.color.r}, ${node.color.g}, ${node.color.b}, ${alpha})`
        ctx.beginPath()
        ctx.arc(node.x * w, node.y * h, r, 0, Math.PI * 2)
        ctx.fill()

        // Bright center
        ctx.fillStyle = `rgba(255, 255, 255, ${alpha * 0.6})`
        ctx.beginPath()
        ctx.arc(node.x * w, node.y * h, r * 0.3, 0, Math.PI * 2)
        ctx.fill()
      }

      // Draw agent hub nodes
      for (const node of nodes) {
        if (!node.isAgent) continue

        const pulse = Math.sin(timeRef.current * 1.5 + node.pulsePhase) * 0.15 + 0.85
        const r = node.radius * pulse

        // Large outer glow
        const gradient = ctx.createRadialGradient(
          node.x * w, node.y * h, 0,
          node.x * w, node.y * h, r * 4,
        )
        gradient.addColorStop(0, `rgba(${node.color.r}, ${node.color.g}, ${node.color.b}, 0.5)`)
        gradient.addColorStop(0.3, `rgba(${node.color.r}, ${node.color.g}, ${node.color.b}, 0.15)`)
        gradient.addColorStop(0.7, `rgba(${node.color.r}, ${node.color.g}, ${node.color.b}, 0.03)`)
        gradient.addColorStop(1, `rgba(${node.color.r}, ${node.color.g}, ${node.color.b}, 0)`)
        ctx.fillStyle = gradient
        ctx.beginPath()
        ctx.arc(node.x * w, node.y * h, r * 4, 0, Math.PI * 2)
        ctx.fill()

        // Core ring
        ctx.strokeStyle = `rgba(${node.color.r}, ${node.color.g}, ${node.color.b}, 0.8)`
        ctx.lineWidth = 2
        ctx.beginPath()
        ctx.arc(node.x * w, node.y * h, r, 0, Math.PI * 2)
        ctx.stroke()

        // Inner fill
        ctx.fillStyle = `rgba(${node.color.r}, ${node.color.g}, ${node.color.b}, 0.2)`
        ctx.beginPath()
        ctx.arc(node.x * w, node.y * h, r, 0, Math.PI * 2)
        ctx.fill()

        // Center bright dot
        ctx.fillStyle = `rgba(${node.color.r}, ${node.color.g}, ${node.color.b}, 0.9)`
        ctx.beginPath()
        ctx.arc(node.x * w, node.y * h, 3, 0, Math.PI * 2)
        ctx.fill()

        // Label
        ctx.fillStyle = `rgba(${node.color.r}, ${node.color.g}, ${node.color.b}, 0.7)`
        ctx.font = '10px monospace'
        ctx.textAlign = 'center'
        ctx.fillText(node.key, node.x * w, node.y * h + r + 14)
      }

      // Update and draw particles
      for (let i = particles.length - 1; i >= 0; i--) {
        const p = particles[i]
        p.x += p.vx
        p.y += p.vy
        p.vx *= 0.995
        p.vy *= 0.995
        p.life -= 1 / p.maxLife

        if (p.life <= 0) {
          particles.splice(i, 1)
          continue
        }

        const alpha = p.life * 0.6
        ctx.fillStyle = `rgba(${p.color.r}, ${p.color.g}, ${p.color.b}, ${alpha})`
        ctx.beginPath()
        ctx.arc(p.x * w, p.y * h, p.size * p.life, 0, Math.PI * 2)
        ctx.fill()
      }

      // Spawn ambient particles periodically
      if (Math.random() < 0.05 && nodes.length > 0) {
        const randomNode = nodes[Math.floor(Math.random() * nodes.length)]
        if (!randomNode.isAgent && randomNode.strength > 0.1) {
          spawnParticles(randomNode, 2)
        }
      }

      animFrameRef.current = requestAnimationFrame(animate)
    }

    animFrameRef.current = requestAnimationFrame(animate)

    return () => {
      window.removeEventListener('resize', resize)
      cancelAnimationFrame(animFrameRef.current)
    }
  }, [spawnParticles])

  return (
    <canvas
      ref={canvasRef}
      className={cn('w-full h-full bg-[#0a0a0e]', className)}
      style={{ display: 'block' }}
    />
  )
}
