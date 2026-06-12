
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
    title: 'Mission Control',
    description: 'Describe a goal and Automatos decomposes it into tasks, assigns specialist agents, executes, verifies, and reports back.',
    gradient: 'from-warning to-red-500',
    features: ['Intelligent Task Decomposition', 'Automated Agent Assignment', 'Built-in Verification', 'Real-time Progress Tracking']
  },
  {
    icon: Brain,
    title: 'Business Knowledge',
    description: 'Upload your documents, connect your data sources. Every agent understands your business context from day one.',
    gradient: 'from-purple-500 to-pink-500',
    features: ['Document Understanding', 'Semantic Search', 'Continuous Learning', 'Context-Aware Agents']
  },
  {
    icon: Workflow,
    title: 'Playbooks',
    description: 'Save successful missions as reusable playbooks. Schedule them to run daily, weekly, or trigger on demand.',
    gradient: 'from-blue-500 to-cyan-500',
    features: ['Save from Missions', 'Scheduled Execution', 'Template Library', 'One-Click Replay']
  },
  {
    icon: Shield,
    title: 'You Stay in Control',
    description: 'Review every plan before it runs. Approve, reject, or redirect. Full audit trail of every decision.',
    gradient: 'from-green-500 to-emerald-500',
    features: ['Plan Approval Gate', 'Human-in-the-Loop Review', 'Complete Audit Trail', 'Workspace Isolation']
  }
]

const cognitiveFunctions = [
  {
    icon: Target,
    title: 'Plan & Decompose',
    description: 'Give a high-level goal and watch it break into verified, sequenced tasks automatically.'
  },
  {
    icon: FileText,
    title: 'Research & Report',
    description: 'Deep research across sources with structured deliverables your team can act on.'
  },
  {
    icon: Code,
    title: 'Build & Automate',
    description: 'Agents that write, test, and deploy code — or process documents, data, and files.'
  },
  {
    icon: GitBranch,
    title: 'Learn & Improve',
    description: 'Every mission makes your agents smarter. History-based scoring surfaces your best performers.'
  }
]

const dashboardFeatures = [
  {
    icon: BarChart3,
    title: 'Mission Analytics',
    description: 'Track success rates, token costs, and agent performance across every mission.',
    color: 'text-blue-400'
  },
  {
    icon: Eye,
    title: 'Live Command Centre',
    description: 'See every agent, every task, every mission in real-time. Know what needs your attention.',
    color: 'text-green-400'
  },
  {
    icon: Database,
    title: 'Knowledge Base',
    description: 'Upload documents, sync cloud drives. Your agents search and reference your business data.',
    color: 'text-purple-400'
  },
  {
    icon: MessageSquare,
    title: 'Agent Reports',
    description: 'Daily standups, task reports, and deliverables from every agent — graded and searchable.',
    color: 'text-warning'
  }
]

const enterpriseFeatures = [
  {
    icon: Globe,
    title: 'Any LLM Provider',
    description: 'OpenRouter, OpenAI, Anthropic, Google, or self-hosted. Mix models per agent.',
    stats: '200+ Models'
  },
  {
    icon: Database,
    title: 'Your Data, Your Control',
    description: 'Workspace isolation, encrypted storage, and full data residency options.',
    stats: 'Multi-Tenant'
  },
  {
    icon: Zap,
    title: 'API-First',
    description: 'Every feature available via REST API. Build custom integrations or use the dashboard.',
    stats: 'Full API'
  },
  {
    icon: Lock,
    title: 'Enterprise Security',
    description: 'SSO, role-based access, comprehensive audit trails, and workspace-level permissions.',
    stats: 'SOC2 Ready'
  }
]

const metrics = [
  { label: 'Minutes to First Mission', value: '5', icon: Zap },
  { label: 'Plan, Execute, Verify', value: 'Auto', icon: Target },
  { label: 'Agents Get Smarter', value: 'Always', icon: Brain },
  { label: 'You Approve Everything', value: '100%', icon: Shield }
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
            The Autonomous Business Operating System
          </Badge>
          <h1 className="text-6xl md:text-7xl font-bold leading-tight">
            Your AI Team That{' '}
            <span className="gradient-text bg-gradient-to-r from-warning via-red-500 to-pink-500">
              Runs Your Business
            </span>
          </h1>
          <p className="text-xl md:text-2xl text-muted-foreground max-w-4xl mx-auto leading-relaxed">
            Tell Automatos what you need done. It plans the work, assigns specialist agents,
            executes, verifies, and reports back. You stay in control.
          </p>
        </motion.div>

        <motion.div variants={itemVariants} className="flex flex-col sm:flex-row gap-4 justify-center">
          <Link href="/agents">
            <Button size="lg" className="text-lg px-8 py-4 bg-gradient-to-r from-warning to-red-500 hover:from-warning hover:to-red-600">
              <Play className="w-5 h-5 mr-2" />
              Start Your First Mission
            </Button>
          </Link>
          <Link href="/assignments?tab=missions">
            <Button variant="outline" size="lg" className="text-lg px-8 py-4">
              <Eye className="w-5 h-5 mr-2" />
              See It In Action
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
                <metric.icon className="w-8 h-8 text-warning" />
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
            <span className="gradient-text">How It Works</span>
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            From goal to deliverable — Automatos handles the planning, execution, and quality control
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
              <Card className="glass-card h-full border-0 shadow-2xl hover:shadow-warning/10">
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
            What Your <span className="gradient-text">Agents Can Do</span>
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Specialist agents that plan, research, build, and learn — so you can focus on decisions
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
              <Card className="glass-card h-full border-0 text-center p-6">
                <div className="w-12 h-12 mx-auto mb-4 rounded-xl bg-gradient-to-br from-warning to-red-500 flex items-center justify-center">
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
            Your <span className="gradient-text">Command Centre</span>
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            See every agent, every task, every mission in real-time. Know what needs your attention.
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
              <Card className="glass-card border-0 p-6">
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
              Open Command Centre
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
            <span className="gradient-text">Built for</span> Production
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Flexible infrastructure that adapts to your stack, your models, and your security requirements
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
              <Card className="glass-card border-0 text-center p-6 h-full">
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
              Ready to Put Your
              <span className="gradient-text block">Business on Autopilot?</span>
            </h2>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto leading-relaxed">
              Upload your business docs, launch your first mission, and watch
              Automatos build your AI team and get to work.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center pt-4">
              <Link href="/agents">
                <Button size="lg" className="text-lg px-12 py-4 bg-gradient-to-r from-warning to-red-500 hover:from-warning hover:to-red-600">
                  <Zap className="w-5 h-5 mr-2" />
                  Start Your First Mission
                </Button>
              </Link>
              <Link href="/assignments?tab=missions">
                <Button variant="outline" size="lg" className="text-lg px-12 py-4">
                  <Workflow className="w-5 h-5 mr-2" />
                  See Missions in Action
                </Button>
              </Link>
            </div>
          </motion.div>
        </div>
      </motion.section>
    </div>
  )
}
