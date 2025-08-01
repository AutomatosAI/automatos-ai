
"""
Professional Agent Database Seeder
==================================

Seeds the database with professional agent types and their associated skills.
Creates sample agents for testing and demonstration purposes.
"""

import asyncio
import sys
import os
from datetime import datetime
from sqlalchemy.orm import Session

# Add the orchestrator directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.database.database import get_db, engine
from src.database.models import Agent, Skill, Base, AgentType, SkillType
from agents import AGENT_REGISTRY

class ProfessionalAgentSeeder:
    """Seeds database with professional agents and skills"""
    
    def __init__(self):
        self.db = next(get_db())
        self.created_agents = []
        self.created_skills = []
    
    def create_tables(self):
        """Create database tables if they don't exist"""
        try:
            Base.metadata.create_all(bind=engine)
            print("✅ Database tables created/verified")
        except Exception as e:
            print(f"❌ Error creating tables: {e}")
            raise
    
    def seed_professional_skills(self):
        """Seed database with professional skills from all agent types"""
        print("🔧 Seeding Professional Skills")
        print("-" * 50)
        
        all_skills = {}
        
        # Collect all unique skills from agent types
        for agent_type, agent_class in AGENT_REGISTRY.items():
            temp_agent = agent_class(
                agent_id=0,
                name="temp",
                description="temp"
            )
            
            for skill_name, skill in temp_agent.skills.items():
                if skill_name not in all_skills:
                    all_skills[skill_name] = skill
        
        # Create skills in database
        for skill_name, skill in all_skills.items():
            try:
                # Check if skill already exists
                existing_skill = self.db.query(Skill).filter(Skill.name == skill_name).first()
                if existing_skill:
                    print(f"⏭️  Skill '{skill_name}' already exists")
                    continue
                
                # Create new skill
                db_skill = Skill(
                    name=skill.name,
                    description=skill.description,
                    skill_type=skill.skill_type.value,
                    implementation="professional_agent_skill",
                    parameters=skill.parameters,
                    performance_data={},
                    created_by="system_seeder"
                )
                
                self.db.add(db_skill)
                self.created_skills.append(skill_name)
                print(f"✅ Created skill: {skill_name}")
                
            except Exception as e:
                print(f"❌ Error creating skill {skill_name}: {e}")
        
        try:
            self.db.commit()
            print(f"💾 Committed {len(self.created_skills)} skills to database")
        except Exception as e:
            self.db.rollback()
            print(f"❌ Error committing skills: {e}")
            raise
    
    def seed_professional_agents(self):
        """Seed database with sample professional agents"""
        print("\n🤖 Seeding Professional Agents")
        print("-" * 50)
        
        agent_configs = [
            {
                "name": "Senior Code Architect",
                "agent_type": "code_architect",
                "description": "Expert in software architecture design, code analysis, and best practices implementation",
                "configuration": {
                    "experience_level": "senior",
                    "specializations": ["microservices", "clean_architecture", "design_patterns"],
                    "preferred_languages": ["python", "javascript", "java", "go"]
                }
            },
            {
                "name": "Cybersecurity Specialist",
                "agent_type": "security_expert",
                "description": "Specialized in application security, vulnerability assessment, and compliance validation",
                "configuration": {
                    "security_frameworks": ["owasp", "nist", "iso27001"],
                    "specializations": ["web_security", "api_security", "cloud_security"],
                    "compliance_expertise": ["gdpr", "hipaa", "pci_dss"]
                }
            },
            {
                "name": "Performance Engineering Expert",
                "agent_type": "performance_optimizer",
                "description": "Expert in system performance analysis, optimization, and scalability planning",
                "configuration": {
                    "optimization_areas": ["application", "database", "infrastructure"],
                    "specializations": ["load_testing", "profiling", "capacity_planning"],
                    "tools_expertise": ["jmeter", "prometheus", "grafana"]
                }
            },
            {
                "name": "Senior Data Scientist",
                "agent_type": "data_analyst",
                "description": "Advanced data analysis, machine learning, and business intelligence specialist",
                "configuration": {
                    "analysis_types": ["descriptive", "predictive", "prescriptive"],
                    "specializations": ["machine_learning", "statistical_analysis", "data_visualization"],
                    "domain_expertise": ["business_intelligence", "customer_analytics", "financial_analysis"]
                }
            },
            {
                "name": "DevOps Infrastructure Lead",
                "agent_type": "infrastructure_manager",
                "description": "Expert in cloud infrastructure, DevOps practices, and system reliability",
                "configuration": {
                    "cloud_platforms": ["aws", "azure", "gcp"],
                    "specializations": ["kubernetes", "terraform", "ci_cd", "monitoring"],
                    "expertise_areas": ["scalability", "reliability", "cost_optimization"]
                }
            },
            {
                "name": "Versatile AI Assistant",
                "agent_type": "custom",
                "description": "Flexible AI assistant capable of handling diverse tasks and adapting to specific requirements",
                "configuration": {
                    "adaptability": "high",
                    "specializations": ["general_purpose", "task_automation", "problem_solving"],
                    "customization_level": "advanced"
                }
            }
        ]
        
        for agent_config in agent_configs:
            try:
                # Check if agent already exists
                existing_agent = self.db.query(Agent).filter(Agent.name == agent_config["name"]).first()
                if existing_agent:
                    print(f"⏭️  Agent '{agent_config['name']}' already exists")
                    continue
                
                # Create new agent
                db_agent = Agent(
                    name=agent_config["name"],
                    description=agent_config["description"],
                    agent_type=agent_config["agent_type"],
                    status="active",
                    configuration=agent_config["configuration"],
                    performance_metrics={
                        "tasks_completed": 0,
                        "success_rate": 0.0,
                        "average_execution_time": 0.0,
                        "total_execution_time": 0.0,
                        "error_count": 0
                    },
                    created_by="system_seeder"
                )
                
                self.db.add(db_agent)
                self.db.flush()  # Get the ID
                
                # Add relevant skills to agent
                agent_type = agent_config["agent_type"]
                if agent_type in AGENT_REGISTRY:
                    temp_agent = AGENT_REGISTRY[agent_type](
                        agent_id=0,
                        name="temp",
                        description="temp"
                    )
                    
                    # Get skills from database that match agent's default skills
                    agent_skill_names = temp_agent.default_skills
                    skills = self.db.query(Skill).filter(Skill.name.in_(agent_skill_names)).all()
                    
                    # Associate skills with agent
                    db_agent.skills.extend(skills)
                    
                    print(f"✅ Created agent: {agent_config['name']} with {len(skills)} skills")
                else:
                    print(f"✅ Created agent: {agent_config['name']} (no skills - unknown type)")
                
                self.created_agents.append(agent_config["name"])
                
            except Exception as e:
                print(f"❌ Error creating agent {agent_config['name']}: {e}")
        
        try:
            self.db.commit()
            print(f"💾 Committed {len(self.created_agents)} agents to database")
        except Exception as e:
            self.db.rollback()
            print(f"❌ Error committing agents: {e}")
            raise
    
    def verify_seeded_data(self):
        """Verify that seeded data is correctly stored"""
        print("\n🔍 Verifying Seeded Data")
        print("-" * 50)
        
        # Count agents by type
        agent_counts = {}
        for agent_type in ["code_architect", "security_expert", "performance_optimizer", 
                          "data_analyst", "infrastructure_manager", "custom"]:
            count = self.db.query(Agent).filter(Agent.agent_type == agent_type).count()
            agent_counts[agent_type] = count
        
        print("Agent counts by type:")
        for agent_type, count in agent_counts.items():
            print(f"  {agent_type}: {count}")
        
        # Count total skills
        total_skills = self.db.query(Skill).count()
        print(f"Total skills in database: {total_skills}")
        
        # Verify agent-skill associations
        agents_with_skills = self.db.query(Agent).filter(Agent.skills.any()).count()
        print(f"Agents with associated skills: {agents_with_skills}")
        
        # Sample agent details
        sample_agent = self.db.query(Agent).filter(Agent.agent_type == "code_architect").first()
        if sample_agent:
            print(f"\nSample agent details:")
            print(f"  Name: {sample_agent.name}")
            print(f"  Type: {sample_agent.agent_type}")
            print(f"  Skills: {len(sample_agent.skills)}")
            print(f"  Status: {sample_agent.status}")
        
        return {
            "total_agents": sum(agent_counts.values()),
            "total_skills": total_skills,
            "agents_with_skills": agents_with_skills,
            "agent_counts": agent_counts
        }
    
    def run_seeding(self):
        """Run complete seeding process"""
        print("🌱 PROFESSIONAL AGENT DATABASE SEEDING")
        print("=" * 80)
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        try:
            # Create tables
            self.create_tables()
            
            # Seed skills first (agents depend on skills)
            self.seed_professional_skills()
            
            # Seed agents
            self.seed_professional_agents()
            
            # Verify seeded data
            verification_results = self.verify_seeded_data()
            
            print("\n" + "=" * 80)
            print("🎉 SEEDING COMPLETED SUCCESSFULLY")
            print("=" * 80)
            print(f"✅ Created {len(self.created_skills)} skills")
            print(f"✅ Created {len(self.created_agents)} agents")
            print(f"✅ Total agents in database: {verification_results['total_agents']}")
            print(f"✅ Total skills in database: {verification_results['total_skills']}")
            print(f"✅ Agents with skills: {verification_results['agents_with_skills']}")
            
            return True
            
        except Exception as e:
            print(f"\n❌ SEEDING FAILED: {e}")
            return False
        
        finally:
            self.db.close()

def main():
    """Main seeding function"""
    seeder = ProfessionalAgentSeeder()
    success = seeder.run_seeding()
    
    if success:
        print("\n🚀 Database is ready for professional agent testing!")
        print("You can now:")
        print("  1. Run the professional agent test suite")
        print("  2. Start the API server and test agent endpoints")
        print("  3. Use the frontend to create and manage professional agents")
    else:
        print("\n💥 Seeding failed. Please check the errors above and try again.")

if __name__ == "__main__":
    main()
