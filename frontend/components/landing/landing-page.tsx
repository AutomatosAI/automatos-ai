
'use client'

import { motion } from 'framer-motion'
import { useInView } from 'react-intersection-observer'
import { 
  Bot, 
  FileText, 
  GitBranch, 
  Activity,
  TrendingUp,
  Clock,
  CheckCircle,
  AlertTriangle,
  Zap,
  Users,
  Database,
  Cpu,
  Brain,
  Shield,
  Layers,
  Network,
  BarChart3,
  Search,
  Workflow,
  Settings,
  Target,
  Sparkles,
  ArrowRight,
  Play,
  Star,
  Code,
  Lock,
  Globe,
  MessageSquare,
  Eye,
  Gauge
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import Link from 'next/link'

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1
    }
  }
}

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.6 }
  }
}

const coreFeatures = [
  {
    icon: Bot,
    title: 'Multi-Agent Orchestration',
    description: 'Advanced AI agents working in perfect harmony with real-time coordination and intelligent task distribution.',
    gradient: 'from-orange-500 to-red-500',
    features: ['Real-time Communication', 'Intelligent Task Routing', 'Performance Monitoring', 'Scalable Architecture']
  },
  {
    icon: Brain,
    title: 'Context Engineering',
    description: 'RAG-powered knowledge system with vector embeddings and semantic search for context-aware intelligence.',
    gradient: 'from-purple-500 to-pink-500',
    features: ['Vector Embeddings', 'Semantic Search', 'Learning Engine', 'Context Awareness']
  },
  {
    icon: Workflow,
    title: 'Visual Workflow Builder',
    description: 'Intuitive drag-and-drop interface for creating complex multi-agent workflows with visual execution tracking.',
    gradient: 'from-blue-500 to-cyan-500',
    features: ['Visual Builder', 'Execution Tracking', 'Template Library', 'Version Control']
  },
  {
    icon: Shield,
    title: 'Enterprise Security',
    description: 'Production-grade security with advanced authentication, authorization, and comprehensive audit trails.',
    gradient: 'from-green-500 to-emerald-500',
    features: ['Authentication', 'Authorization', 'Audit Trails', 'Data Encryption']
  }
]

const cognitiveFunctions = [
  {
    icon: Target,
    title: 'Task Breakdown',
    description: 'Intelligent decomposition of complex objectives into manageable, actionable tasks.'
  },
  {
    icon: FileText,
    title: 'Content Generation',
    description: 'Advanced content creation capabilities with context-aware generation and optimization.'
  },
  {
    icon: Code,
    title: 'File Operations',
    description: 'Comprehensive file management, processing, and transformation capabilities.'
  },
  {
    icon: GitBranch,
    title: 'Git Integration',
    description: 'Seamless version control integration with automated commit and collaboration features.'
  }
]

const dashboardFeatures = [
  {
    icon: BarChart3,
    title: 'Performance Analytics',
    description: 'Real-time insights into agent performance, workflow execution, and system health.',
    color: 'text-blue-400'
  },
  {
    icon: Eye,
    title: 'Real-time Monitoring',
    description: 'Live system monitoring with alerts, logging, and comprehensive visibility.',
    color: 'text-green-400'
  },
  {
    icon: Database,
    title: 'Document Management',
    description: 'Advanced document processing with OCR, extraction, and knowledge indexing.',
    color: 'text-purple-400'
  },
  {
    icon: MessageSquare,
    title: 'Agent Communication',
    description: 'Visualize and manage inter-agent communication patterns and coordination.',
    color: 'text-orange-400'
  }
]

const enterpriseFeatures = [
  {
    icon: Globe,
    title: 'Production Ready',
    description: 'Docker deployment, microservices architecture, and enterprise scalability.',
    stats: '99.9% Uptime'
  },
  {
    icon: Database,
    title: 'PostgreSQL + pgvector',
    description: 'Robust database with vector extensions for high-performance semantic search.',
    stats: '1M+ Vectors'
  },
  {
    icon: Zap,
    title: '25+ API Endpoints',
    description: 'Comprehensive RESTful API with extensive documentation and SDKs.',
    stats: '100ms Response'
  },
  {
    icon: Lock,
    title: 'Advanced Security',
    description: 'Enterprise-grade authentication, authorization, and data protection.',
    stats: 'SOC2 Ready'
  }
]

const metrics = [
  { label: 'Active Agents', value: '24+', icon: Bot },
  { label: 'Workflows Executed', value: '10K+', icon: Workflow },
  { label: 'Documents Processed', value: '50K+', icon: FileText },
  { label: 'API Calls/Day', value: '1M+', icon: Zap }
]

export function LandingPage() {
  const [heroRef, heroInView] = useInView({ triggerOnce: true, threshold: 0.1 })
  const [featuresRef, featuresInView] = useInView({ triggerOnce: true, threshold: 0.1 })
  const [cognitiveRef, cognitiveInView] = useInView({ triggerOnce: true, threshold: 0.1 })
  const [dashboardRef, dashboardInView] = useInView({ triggerOnce: true, threshold: 0.1 })
  const [enterpriseRef, enterpriseInView] = useInView({ triggerOnce: true, threshold: 0.1 })
  const [ctaRef, ctaInView] = useInView({ triggerOnce: true, threshold: 0.1 })

  return (
    <div className="space-y-20">
      {/* Hero Section */}
      <motion.section
        ref={heroRef}
        className="text-center space-y-8 py-20"
        variants={containerVariants}
        initial="hidden"
        animate={heroInView ? "visible" : "hidden"}
      >
        <motion.div variants={itemVariants} className="space-y-4">
          <Badge variant="secondary" className="px-4 py-2 text-sm">
            <Sparkles className="w-4 h-4 mr-2" />
            World-Class Enterprise AI Platform
          </Badge>
          <h1 className="text-6xl md:text-7xl font-bold leading-tight">
            Welcome to{' '}
            <span className="gradient-text bg-gradient-to-r from-orange-500 via-red-500 to-pink-500">
              Automotas AI
            </span>
          </h1>
          <p className="text-xl md:text-2xl text-muted-foreground max-w-4xl mx-auto leading-relaxed">
            The most advanced multi-agent orchestration platform with intelligent coordination, 
            context engineering, and beautiful enterprise dashboards. Built for production scale.
          </p>
        </motion.div>

        <motion.div variants={itemVariants} className="flex flex-col sm:flex-row gap-4 justify-center">
          <Link href="/agents">
            <Button size="lg" className="text-lg px-8 py-4 bg-gradient-to-r from-orange-500 to-red-500 hover:from-orange-600 hover:to-red-600">
              <Play className="w-5 h-5 mr-2" />
              Launch Platform
            </Button>
          </Link>
          <Link href="/documents">
            <Button variant="outline" size="lg" className="text-lg px-8 py-4">
              <Eye className="w-5 h-5 mr-2" />
              View Dashboard
            </Button>
          </Link>
        </motion.div>

        {/* Metrics */}
        <motion.div
          variants={itemVariants}
          className="grid grid-cols-2 md:grid-cols-4 gap-8 max-w-4xl mx-auto mt-16"
        >
          {metrics.map((metric, index) => (
            <motion.div
              key={metric.label}
              className="text-center space-y-2"
              initial={{ opacity: 0, scale: 0.8 }}
              animate={heroInView ? { opacity: 1, scale: 1 } : {}}
              transition={{ duration: 0.6, delay: index * 0.1 }}
            >
              <div className="flex justify-center mb-2">
                <metric.icon className="w-8 h-8 text-orange-500" />
              </div>
              <div className="text-3xl font-bold">{metric.value}</div>
              <div className="text-sm text-muted-foreground">{metric.label}</div>
            </motion.div>
          ))}
        </motion.div>
      </motion.section>

      {/* Core Features */}
      <motion.section
        ref={featuresRef}
        className="space-y-12"
        variants={containerVariants}
        initial="hidden"
        animate={featuresInView ? "visible" : "hidden"}
      >
        <motion.div variants={itemVariants} className="text-center space-y-4">
          <h2 className="text-4xl md:text-5xl font-bold">
            <span className="gradient-text">World-Class Features</span>
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Enterprise-grade capabilities that set new standards for AI orchestration platforms
          </p>
        </motion.div>

        <motion.div
          variants={containerVariants}
          className="grid grid-cols-1 md:grid-cols-2 gap-8"
        >
          {coreFeatures.map((feature, index) => (
            <motion.div
              key={feature.title}
              variants={itemVariants}
              whileHover={{ scale: 1.02, y: -5 }}
              transition={{ duration: 0.3 }}
            >
              <Card className="glass-card h-full border-0 shadow-2xl hover:shadow-orange-500/10">
                <CardHeader className="space-y-4">
                  <div className={`w-16 h-16 rounded-2xl bg-gradient-to-br ${feature.gradient} flex items-center justify-center`}>
                    <feature.icon className="w-8 h-8 text-white" />
                  </div>
                  <div>
                    <CardTitle className="text-2xl mb-2">{feature.title}</CardTitle>
                    <CardDescription className="text-base leading-relaxed">
                      {feature.description}
                    </CardDescription>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    {feature.features.map((item, idx) => (
                      <div key={idx} className="flex items-center text-sm text-muted-foreground">
                        <CheckCircle className="w-4 h-4 mr-2 text-green-500" />
                        {item}
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </motion.div>
      </motion.section>

      {/* Cognitive Functions */}
      <motion.section
        ref={cognitiveRef}
        className="space-y-12"
        variants={containerVariants}
        initial="hidden"
        animate={cognitiveInView ? "visible" : "hidden"}
      >
        <motion.div variants={itemVariants} className="text-center space-y-4">
          <h2 className="text-4xl md:text-5xl font-bold">
            Advanced <span className="gradient-text">Cognitive Functions</span>
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Four core AI capabilities that power intelligent orchestration and automation
          </p>
        </motion.div>

        <motion.div
          variants={containerVariants}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
        >
          {cognitiveFunctions.map((func, index) => (
            <motion.div
              key={func.title}
              variants={itemVariants}
              whileHover={{ scale: 1.05, rotateY: 5 }}
              transition={{ duration: 0.3 }}
            >
              <Card className="glass-card h-full border-0 text-center p-6 hover:shadow-2xl hover:shadow-orange-500/20">
                <div className="w-12 h-12 mx-auto mb-4 rounded-xl bg-gradient-to-br from-orange-500 to-red-500 flex items-center justify-center">
                  <func.icon className="w-6 h-6 text-white" />
                </div>
                <h3 className="text-lg font-semibold mb-2">{func.title}</h3>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  {func.description}
                </p>
              </Card>
            </motion.div>
          ))}
        </motion.div>
      </motion.section>

      {/* Dashboard Features */}
      <motion.section
        ref={dashboardRef}
        className="space-y-12"
        variants={containerVariants}
        initial="hidden"
        animate={dashboardInView ? "visible" : "hidden"}
      >
        <motion.div variants={itemVariants} className="text-center space-y-4">
          <h2 className="text-4xl md:text-5xl font-bold">
            Beautiful <span className="gradient-text">Management Dashboards</span>
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Stunning, intuitive interfaces that make complex AI orchestration simple and visual
          </p>
        </motion.div>

        <motion.div
          variants={containerVariants}
          className="grid grid-cols-1 md:grid-cols-2 gap-8"
        >
          {dashboardFeatures.map((feature, index) => (
            <motion.div
              key={feature.title}
              variants={itemVariants}
              whileHover={{ scale: 1.02 }}
              transition={{ duration: 0.3 }}
            >
              <Card className="glass-card border-0 p-6 hover:shadow-xl">
                <div className="flex items-start space-x-4">
                  <div className={`w-12 h-12 rounded-xl bg-black/20 flex items-center justify-center`}>
                    <feature.icon className={`w-6 h-6 ${feature.color}`} />
                  </div>
                  <div className="flex-1">
                    <h3 className="text-xl font-semibold mb-2">{feature.title}</h3>
                    <p className="text-muted-foreground leading-relaxed">
                      {feature.description}
                    </p>
                  </div>
                </div>
              </Card>
            </motion.div>
          ))}
        </motion.div>

        <motion.div variants={itemVariants} className="text-center">
          <Link href="/analytics">
            <Button size="lg" variant="outline" className="px-8">
              <BarChart3 className="w-5 h-5 mr-2" />
              Explore Analytics Dashboard
              <ArrowRight className="w-5 h-5 ml-2" />
            </Button>
          </Link>
        </motion.div>
      </motion.section>

      {/* Enterprise Features */}
      <motion.section
        ref={enterpriseRef}
        className="space-y-12"
        variants={containerVariants}
        initial="hidden"
        animate={enterpriseInView ? "visible" : "hidden"}
      >
        <motion.div variants={itemVariants} className="text-center space-y-4">
          <h2 className="text-4xl md:text-5xl font-bold">
            <span className="gradient-text">Enterprise-Ready</span> Architecture
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Production-grade infrastructure built for scale, security, and reliability
          </p>
        </motion.div>

        <motion.div
          variants={containerVariants}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
        >
          {enterpriseFeatures.map((feature, index) => (
            <motion.div
              key={feature.title}
              variants={itemVariants}
              whileHover={{ scale: 1.05, y: -10 }}
              transition={{ duration: 0.3 }}
            >
              <Card className="glass-card border-0 text-center p-6 h-full hover:shadow-2xl hover:shadow-green-500/20">
                <div className="w-12 h-12 mx-auto mb-4 rounded-xl bg-gradient-to-br from-green-500 to-emerald-500 flex items-center justify-center">
                  <feature.icon className="w-6 h-6 text-white" />
                </div>
                <h3 className="text-lg font-semibold mb-2">{feature.title}</h3>
                <p className="text-sm text-muted-foreground mb-4 leading-relaxed">
                  {feature.description}
                </p>
                <Badge variant="secondary" className="text-xs">
                  {feature.stats}
                </Badge>
              </Card>
            </motion.div>
          ))}
        </motion.div>
      </motion.section>

      {/* Call to Action */}
      <motion.section
        ref={ctaRef}
        className="text-center space-y-8 py-20"
        initial={{ opacity: 0, y: 50 }}
        animate={ctaInView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.8 }}
      >
        <div className="glass-card p-12 max-w-4xl mx-auto border-0 shadow-2xl">
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={ctaInView ? { opacity: 1, scale: 1 } : {}}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="space-y-6"
          >
            <h2 className="text-4xl md:text-5xl font-bold">
              Ready to Transform Your
              <span className="gradient-text block">AI Operations?</span>
            </h2>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto leading-relaxed">
              Join the future of AI orchestration with our world-class platform. 
              Start building intelligent multi-agent workflows today.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center pt-4">
              <Link href="/agents">
                <Button size="lg" className="text-lg px-12 py-4 bg-gradient-to-r from-orange-500 to-red-500 hover:from-orange-600 hover:to-red-600">
                  <Zap className="w-5 h-5 mr-2" />
                  Get Started Now
                </Button>
              </Link>
              <Link href="/workflows">
                <Button variant="outline" size="lg" className="text-lg px-12 py-4">
                  <Workflow className="w-5 h-5 mr-2" />
                  Explore Workflows
                </Button>
              </Link>
            </div>
          </motion.div>
        </div>
      </motion.section>
    </div>
  )
}
