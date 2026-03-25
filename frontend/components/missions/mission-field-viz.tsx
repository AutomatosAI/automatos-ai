'use client'

import { useRef, useMemo } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { Float, Text, Billboard, OrbitControls } from '@react-three/drei'
import { EffectComposer, Bloom, ChromaticAberration } from '@react-three/postprocessing'
import * as THREE from 'three'
import { cn } from '@/lib/utils'
import type { FieldPattern } from '@/hooks/use-missions-api'

interface MissionFieldVizProps {
  patterns: FieldPattern[]
  className?: string
}

// Agent color palette — vibrant neon for dark backgrounds
const PALETTE = [
  new THREE.Color(0.23, 0.51, 0.96),  // blue
  new THREE.Color(0.06, 0.73, 0.51),  // emerald
  new THREE.Color(0.66, 0.33, 0.97),  // purple
  new THREE.Color(0.96, 0.62, 0.04),  // amber
  new THREE.Color(0.96, 0.25, 0.37),  // rose
  new THREE.Color(0.02, 0.71, 0.83),  // cyan
  new THREE.Color(0.98, 0.45, 0.09),  // orange
  new THREE.Color(0.52, 0.80, 0.09),  // lime
]

function getAgentColor(index: number) {
  return PALETTE[index % PALETTE.length]
}

// ── Central Qdrant Core ──────────────────────────────────────
function QdrantCore() {
  const meshRef = useRef<THREE.Mesh>(null)
  const glowRef = useRef<THREE.Mesh>(null)
  const ringRef = useRef<THREE.Mesh>(null)
  const ring2Ref = useRef<THREE.Mesh>(null)

  useFrame(({ clock }) => {
    const t = clock.getElapsedTime()
    if (meshRef.current) {
      meshRef.current.rotation.y = t * 0.15
      meshRef.current.rotation.x = Math.sin(t * 0.1) * 0.1
    }
    if (glowRef.current) {
      const scale = 1.8 + Math.sin(t * 0.8) * 0.15
      glowRef.current.scale.setScalar(scale)
    }
    if (ringRef.current) {
      ringRef.current.rotation.x = Math.PI / 2 + Math.sin(t * 0.3) * 0.2
      ringRef.current.rotation.z = t * 0.4
    }
    if (ring2Ref.current) {
      ring2Ref.current.rotation.x = Math.PI / 3 + Math.cos(t * 0.2) * 0.15
      ring2Ref.current.rotation.z = -t * 0.3
    }
  })

  return (
    <group>
      {/* Inner icosahedron — the "brain" */}
      <mesh ref={meshRef}>
        <icosahedronGeometry args={[0.6, 1]} />
        <meshStandardMaterial
          color="#6366f1"
          emissive="#6366f1"
          emissiveIntensity={0.8}
          wireframe
          transparent
          opacity={0.7}
        />
      </mesh>

      {/* Solid inner core */}
      <mesh>
        <sphereGeometry args={[0.35, 32, 32]} />
        <meshStandardMaterial
          color="#818cf8"
          emissive="#818cf8"
          emissiveIntensity={1.2}
          transparent
          opacity={0.6}
        />
      </mesh>

      {/* Outer glow sphere */}
      <mesh ref={glowRef}>
        <sphereGeometry args={[0.7, 32, 32]} />
        <meshStandardMaterial
          color="#6366f1"
          emissive="#a78bfa"
          emissiveIntensity={0.5}
          transparent
          opacity={0.08}
          side={THREE.BackSide}
        />
      </mesh>

      {/* Orbital ring 1 */}
      <mesh ref={ringRef}>
        <torusGeometry args={[1.0, 0.008, 16, 100]} />
        <meshStandardMaterial
          color="#818cf8"
          emissive="#818cf8"
          emissiveIntensity={2}
          transparent
          opacity={0.4}
        />
      </mesh>

      {/* Orbital ring 2 */}
      <mesh ref={ring2Ref}>
        <torusGeometry args={[1.2, 0.006, 16, 100]} />
        <meshStandardMaterial
          color="#a78bfa"
          emissive="#a78bfa"
          emissiveIntensity={1.5}
          transparent
          opacity={0.25}
        />
      </mesh>

      {/* Label */}
      <Billboard position={[0, -1.1, 0]}>
        <Text
          fontSize={0.15}
          color="#a78bfa"
          anchorX="center"
          anchorY="top"
        >
          SHARED FIELD
        </Text>
      </Billboard>
    </group>
  )
}

// ── Memory Particle Stream ───────────────────────────────────
interface ParticleStreamProps {
  from: THREE.Vector3
  to: THREE.Vector3
  color: THREE.Color
  count: number
  speed: number
  reverse?: boolean
}

function ParticleStream({ from, to, color, count, speed, reverse }: ParticleStreamProps) {
  const pointsRef = useRef<THREE.Points>(null)
  const positionsRef = useRef<Float32Array | null>(null)
  const progressRef = useRef<Float32Array | null>(null)

  const { positions, progress } = useMemo(() => {
    const positions = new Float32Array(count * 3)
    const progress = new Float32Array(count)
    for (let i = 0; i < count; i++) {
      progress[i] = Math.random()
      // Initialize along path
      const t = progress[i]
      const src = reverse ? to : from
      const dst = reverse ? from : to
      positions[i * 3] = src.x + (dst.x - src.x) * t + (Math.random() - 0.5) * 0.1
      positions[i * 3 + 1] = src.y + (dst.y - src.y) * t + (Math.random() - 0.5) * 0.1
      positions[i * 3 + 2] = src.z + (dst.z - src.z) * t + (Math.random() - 0.5) * 0.1
    }
    return { positions, progress }
  }, [from, to, count, reverse])

  positionsRef.current = positions
  progressRef.current = progress

  useFrame((_, delta) => {
    if (!pointsRef.current || !positionsRef.current || !progressRef.current) return
    const pos = positionsRef.current
    const prog = progressRef.current
    const src = reverse ? to : from
    const dst = reverse ? from : to

    for (let i = 0; i < count; i++) {
      prog[i] += delta * speed * (0.5 + Math.random() * 0.5)
      if (prog[i] > 1) prog[i] = 0

      const t = prog[i]
      // Curved path with slight arc
      const mid = new THREE.Vector3().lerpVectors(src, dst, 0.5)
      mid.y += 0.5

      const p1 = new THREE.Vector3().lerpVectors(src, mid, t)
      const p2 = new THREE.Vector3().lerpVectors(mid, dst, t)
      const point = new THREE.Vector3().lerpVectors(p1, p2, t)

      pos[i * 3] = point.x + (Math.random() - 0.5) * 0.03
      pos[i * 3 + 1] = point.y + (Math.random() - 0.5) * 0.03
      pos[i * 3 + 2] = point.z + (Math.random() - 0.5) * 0.03
    }

    const geo = pointsRef.current.geometry
    geo.attributes.position.needsUpdate = true
  })

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          args={[positions, 3]}
        />
      </bufferGeometry>
      <pointsMaterial
        size={0.04}
        color={color}
        transparent
        opacity={0.8}
        blending={THREE.AdditiveBlending}
        depthWrite={false}
        sizeAttenuation
      />
    </points>
  )
}

// ── Agent Orb ────────────────────────────────────────────────
interface AgentOrbProps {
  agentId: number
  index: number
  totalAgents: number
  patternCount: number
  activeStrength: number
}

function AgentOrb({ agentId, index, totalAgents, patternCount, activeStrength }: AgentOrbProps) {
  const groupRef = useRef<THREE.Group>(null)
  const meshRef = useRef<THREE.Mesh>(null)
  const color = getAgentColor(index)

  const orbitRadius = 3.0 + (index % 2) * 0.8
  const baseAngle = (index / Math.max(totalAgents, 1)) * Math.PI * 2

  const position = useMemo(() => {
    return new THREE.Vector3(
      Math.cos(baseAngle) * orbitRadius,
      (Math.random() - 0.5) * 1.5,
      Math.sin(baseAngle) * orbitRadius,
    )
  }, [baseAngle, orbitRadius])

  useFrame(({ clock }) => {
    if (!groupRef.current) return
    const t = clock.getElapsedTime()
    const angle = baseAngle + t * 0.08 * (1 + index * 0.02)
    groupRef.current.position.x = Math.cos(angle) * orbitRadius
    groupRef.current.position.z = Math.sin(angle) * orbitRadius
    groupRef.current.position.y = Math.sin(t * 0.3 + index) * 0.6

    // Update position ref for particle streams
    position.copy(groupRef.current.position)

    if (meshRef.current) {
      const pulse = 1 + Math.sin(t * 2 + index * 1.5) * 0.1
      meshRef.current.scale.setScalar(pulse)
    }
  })

  const orbSize = 0.25 + Math.min(patternCount * 0.03, 0.3)
  const label = agentId === 0 ? 'System' : `Agent ${agentId}`

  return (
    <group ref={groupRef} position={position}>
      {/* Agent sphere */}
      <Float speed={2} rotationIntensity={0.3} floatIntensity={0.2}>
        <mesh ref={meshRef}>
          <sphereGeometry args={[orbSize, 32, 32]} />
          <meshStandardMaterial
            color={color}
            emissive={color}
            emissiveIntensity={1.5 + activeStrength}
            transparent
            opacity={0.85}
          />
        </mesh>

        {/* Glow halo */}
        <mesh>
          <sphereGeometry args={[orbSize * 2, 16, 16]} />
          <meshStandardMaterial
            color={color}
            emissive={color}
            emissiveIntensity={0.3}
            transparent
            opacity={0.06}
            side={THREE.BackSide}
          />
        </mesh>
      </Float>

      {/* Label */}
      <Billboard position={[0, -orbSize - 0.3, 0]}>
        <Text
          fontSize={0.12}
          color={`#${color.getHexString()}`}
          anchorX="center"
          anchorY="top"
        >
          {label}
        </Text>
        <Text
          fontSize={0.08}
          color="#666"
          anchorX="center"
          anchorY="top"
          position={[0, -0.15, 0]}
        >
          {patternCount} patterns
        </Text>
      </Billboard>

      {/* Particle stream: agent -> core (injection) */}
      <ParticleStream
        from={position}
        to={new THREE.Vector3(0, 0, 0)}
        color={color}
        count={Math.max(8, patternCount * 2)}
        speed={0.3 + activeStrength * 0.2}
      />

      {/* Particle stream: core -> agent (queries) */}
      <ParticleStream
        from={new THREE.Vector3(0, 0, 0)}
        to={position}
        color={new THREE.Color().copy(color).multiplyScalar(0.6)}
        count={Math.max(4, patternCount)}
        speed={0.2}
        reverse
      />
    </group>
  )
}

// ── Pattern Nodes (orbiting their agent) ─────────────────────
interface PatternNodeProps {
  pattern: FieldPattern
  agentPosition: THREE.Vector3
  localIndex: number
  totalLocal: number
  agentColorIndex: number
}

function PatternNode({ pattern, agentPosition, localIndex, totalLocal, agentColorIndex }: PatternNodeProps) {
  const meshRef = useRef<THREE.Mesh>(null)
  const color = getAgentColor(agentColorIndex)

  const orbitRadius = 0.6 + pattern.decayed_strength * 0.4
  const baseAngle = (localIndex / Math.max(totalLocal, 1)) * Math.PI * 2

  useFrame(({ clock }) => {
    if (!meshRef.current) return
    const t = clock.getElapsedTime()
    const angle = baseAngle + t * 0.5
    meshRef.current.position.x = agentPosition.x + Math.cos(angle) * orbitRadius
    meshRef.current.position.y = agentPosition.y + Math.sin(t * 0.8 + localIndex) * 0.15
    meshRef.current.position.z = agentPosition.z + Math.sin(angle) * orbitRadius

    const pulse = 1 + Math.sin(t * 3 + localIndex * 2) * 0.15
    meshRef.current.scale.setScalar(pulse)
  })

  const size = 0.04 + pattern.decayed_strength * 0.08

  return (
    <mesh ref={meshRef}>
      <sphereGeometry args={[size, 16, 16]} />
      <meshStandardMaterial
        color={color}
        emissive={color}
        emissiveIntensity={0.5 + pattern.decayed_strength * 2}
        transparent
        opacity={pattern.is_archived ? 0.2 : 0.7 + pattern.decayed_strength * 0.3}
      />
    </mesh>
  )
}

// ── Background Stars ─────────────────────────────────────────
function StarField() {
  const count = 500
  const positions = useMemo(() => {
    const pos = new Float32Array(count * 3)
    for (let i = 0; i < count; i++) {
      const r = 15 + Math.random() * 20
      const theta = Math.random() * Math.PI * 2
      const phi = Math.acos(2 * Math.random() - 1)
      pos[i * 3] = r * Math.sin(phi) * Math.cos(theta)
      pos[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta)
      pos[i * 3 + 2] = r * Math.cos(phi)
    }
    return pos
  }, [])

  return (
    <points>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          args={[positions, 3]}
        />
      </bufferGeometry>
      <pointsMaterial
        size={0.05}
        color="#4444aa"
        transparent
        opacity={0.5}
        sizeAttenuation
      />
    </points>
  )
}

// ── Grid Floor ───────────────────────────────────────────────
function GridFloor() {
  return (
    <gridHelper
      args={[20, 40, '#1a1a3a', '#0d0d1f']}
      position={[0, -3, 0]}
      rotation={[0, 0, 0]}
    />
  )
}

// ── Scene Composition ────────────────────────────────────────
interface FieldSceneProps {
  patterns: FieldPattern[]
}

function FieldScene({ patterns }: FieldSceneProps) {
  const uniqueAgents = useMemo(() => {
    return [...new Set(patterns.map(p => p.agent_id))]
  }, [patterns])

  const agentIndexMap = useMemo(() => {
    return new Map(uniqueAgents.map((id, i) => [id, i]))
  }, [uniqueAgents])

  // Agent positions tracked via refs in AgentOrb, approximate here for PatternNode
  const agentPositions = useMemo(() => {
    const map = new Map<number, THREE.Vector3>()
    uniqueAgents.forEach((id, i) => {
      const angle = (i / Math.max(uniqueAgents.length, 1)) * Math.PI * 2
      const r = 3.0 + (i % 2) * 0.8
      map.set(id, new THREE.Vector3(Math.cos(angle) * r, 0, Math.sin(angle) * r))
    })
    return map
  }, [uniqueAgents])

  return (
    <>
      {/* Lighting */}
      <ambientLight intensity={0.15} />
      <pointLight position={[0, 0, 0]} intensity={2} color="#6366f1" distance={10} />
      <pointLight position={[5, 5, 5]} intensity={0.3} color="#ffffff" />
      <pointLight position={[-5, -3, -5]} intensity={0.2} color="#a78bfa" />

      {/* Camera controls */}
      <OrbitControls
        enablePan={false}
        minDistance={3}
        maxDistance={12}
        autoRotate
        autoRotateSpeed={0.3}
        maxPolarAngle={Math.PI * 0.75}
        minPolarAngle={Math.PI * 0.25}
      />

      {/* Background */}
      <StarField />
      <GridFloor />
      <fog attach="fog" args={['#0a0a12', 8, 25]} />

      {/* Central Qdrant / Field core */}
      <QdrantCore />

      {/* Agent orbs */}
      {uniqueAgents.map((agentId, i) => {
        const agentPatterns = patterns.filter(p => p.agent_id === agentId)
        const avgStrength = agentPatterns.length > 0
          ? agentPatterns.reduce((s, p) => s + p.decayed_strength, 0) / agentPatterns.length
          : 0

        return (
          <AgentOrb
            key={agentId}
            agentId={agentId}
            index={i}
            totalAgents={uniqueAgents.length}
            patternCount={agentPatterns.length}
            activeStrength={avgStrength}
          />
        )
      })}

      {/* Pattern nodes orbiting their agents */}
      {patterns.slice(0, 60).map((pattern, i) => {
        const agentIdx = agentIndexMap.get(pattern.agent_id) ?? 0
        const agentPos = agentPositions.get(pattern.agent_id) ?? new THREE.Vector3()
        const patternsForAgent = patterns.filter(p => p.agent_id === pattern.agent_id)
        const localIdx = patternsForAgent.indexOf(pattern)

        return (
          <PatternNode
            key={pattern.id}
            pattern={pattern}
            agentPosition={agentPos}
            localIndex={localIdx}
            totalLocal={patternsForAgent.length}
            agentColorIndex={agentIdx}
          />
        )
      })}

      {/* Postprocessing */}
      <EffectComposer>
        <Bloom
          luminanceThreshold={0.2}
          luminanceSmoothing={0.9}
          intensity={1.5}
          mipmapBlur
        />
        <ChromaticAberration
          offset={new THREE.Vector2(0.0005, 0.0005)}
          radialModulation={false}
          modulationOffset={0}
        />
      </EffectComposer>
    </>
  )
}

// ── Main Export ───────────────────────────────────────────────
export function MissionFieldViz({ patterns, className }: MissionFieldVizProps) {
  return (
    <div className={cn('w-full h-full', className)}>
      <Canvas
        camera={{ position: [0, 3, 7], fov: 50 }}
        gl={{
          antialias: true,
          alpha: false,
          powerPreference: 'high-performance',
        }}
        style={{ background: '#0a0a12' }}
        dpr={[1, 2]}
      >
        <FieldScene patterns={patterns} />
      </Canvas>
    </div>
  )
}
