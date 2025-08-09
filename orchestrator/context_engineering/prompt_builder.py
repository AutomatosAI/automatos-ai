
"""
Context-Aware Prompt Builder
============================

Advanced prompt engineering with context injection and dynamic template selection.
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from .context_retriever import ContextResult

logger = logging.getLogger(__name__)

class PromptType(Enum):
    """Types of prompts for different tasks"""
    CODE_GENERATION = "code_generation"
    CODE_ANALYSIS = "code_analysis"
    DOCUMENTATION = "documentation"
    DEBUGGING = "debugging"
    API_DEVELOPMENT = "api_development"
    TESTING = "testing"
    ARCHITECTURE = "architecture"
    OPTIMIZATION = "optimization"
    GENERAL = "general"

@dataclass
class PromptTemplate:
    """Template for prompt generation"""
    name: str
    prompt_type: PromptType
    system_prompt: str
    user_template: str
    context_template: str
    max_context_length: int = 4000
    required_context_types: List[str] = None
    success_rate: float = 0.0
    usage_count: int = 0

class ContextAwarePromptBuilder:
    """Advanced prompt builder with context awareness and template management"""
    
    def __init__(self):
        self.templates = {}
        self.context_strategies = {}
        self.prompt_history = []
        self.success_patterns = {}
        
        self._initialize_templates()
        self._initialize_context_strategies()
    
    def _initialize_templates(self):
        """Initialize prompt templates for different task types"""
        
        # Code Generation Template
        self.templates[PromptType.CODE_GENERATION] = PromptTemplate(
            name="code_generation",
            prompt_type=PromptType.CODE_GENERATION,
            system_prompt="""You are an expert software developer with deep knowledge of multiple programming languages and best practices. 

Your task is to generate high-quality, production-ready code based on the requirements and context provided. 

Key principles:
- Write clean, maintainable, and well-documented code
- Follow language-specific best practices and conventions
- Include proper error handling and edge case management
- Consider security implications and implement secure coding practices
- Write testable code with clear separation of concerns
- Use appropriate design patterns when beneficial

Context will be provided to help you understand existing patterns, APIs, and architectural decisions.""",
            
            user_template="""Generate code for the following requirement:

**Requirement:** {task_description}

**Target Language:** {language}
**Framework/Library:** {framework}

{context_section}

Please provide:
1. Complete, working code implementation
2. Brief explanation of key design decisions
3. Any dependencies or setup requirements
4. Basic usage example if applicable

Code:""",
            
            context_template="""**Relevant Context:**

{context_items}

**Patterns and Examples:**
{patterns}""",
            
            max_context_length=4000,
            required_context_types=['code', 'documentation']
        )
        
        # Code Analysis Template
        self.templates[PromptType.CODE_ANALYSIS] = PromptTemplate(
            name="code_analysis",
            prompt_type=PromptType.CODE_ANALYSIS,
            system_prompt="""You are a senior code reviewer and software architect with expertise in code quality, security, and performance analysis.

Your task is to analyze code thoroughly and provide actionable insights.

Focus areas:
- Code quality and maintainability
- Security vulnerabilities and best practices
- Performance implications and optimizations
- Design patterns and architectural concerns
- Testing coverage and testability
- Documentation and code clarity""",
            
            user_template="""Analyze the following code:

**Code to Analyze:**
```{language}
{code_content}
```

**Analysis Focus:** {analysis_focus}

{context_section}

Please provide:
1. Overall code quality assessment
2. Specific issues and recommendations
3. Security considerations
4. Performance implications
5. Suggested improvements with examples

Analysis:""",
            
            context_template="""**Related Code Context:**

{context_items}

**Best Practices Reference:**
{patterns}""",
            
            max_context_length=3000,
            required_context_types=['code', 'documentation']
        )
        
        # API Development Template
        self.templates[PromptType.API_DEVELOPMENT] = PromptTemplate(
            name="api_development",
            prompt_type=PromptType.API_DEVELOPMENT,
            system_prompt="""You are an expert API developer with extensive experience in RESTful services, GraphQL, authentication, and API security.

Your expertise includes:
- RESTful API design principles
- Authentication and authorization (JWT, OAuth, API keys)
- Data validation and serialization
- Error handling and status codes
- API documentation and OpenAPI/Swagger
- Rate limiting and security best practices
- Database integration and ORM usage""",
            
            user_template="""Develop an API for the following specification:

**API Requirement:** {task_description}

**Framework:** {framework}
**Database:** {database}
**Authentication:** {auth_method}

{context_section}

Please provide:
1. Complete API implementation
2. Authentication/authorization setup
3. Data models and validation
4. Error handling
5. Basic API documentation
6. Example requests/responses

Implementation:""",
            
            context_template="""**API Context and Examples:**

{context_items}

**Authentication Patterns:**
{auth_patterns}

**Database Patterns:**
{db_patterns}""",
            
            max_context_length=4500,
            required_context_types=['code', 'api', 'documentation']
        )
        
        # Debugging Template
        self.templates[PromptType.DEBUGGING] = PromptTemplate(
            name="debugging",
            prompt_type=PromptType.DEBUGGING,
            system_prompt="""You are an expert debugger with deep knowledge of common programming errors, debugging techniques, and problem-solving methodologies.

Your approach:
- Systematic analysis of error symptoms
- Root cause identification
- Step-by-step debugging methodology
- Prevention strategies for similar issues
- Performance and memory considerations
- Tool recommendations for debugging""",
            
            user_template="""Debug the following issue:

**Problem Description:** {problem_description}

**Error Message/Symptoms:**
```
{error_details}
```

**Code Context:**
```{language}
{code_content}
```

{context_section}

Please provide:
1. Root cause analysis
2. Step-by-step debugging approach
3. Specific fix with explanation
4. Prevention strategies
5. Testing recommendations

Solution:""",
            
            context_template="""**Related Issues and Solutions:**

{context_items}

**Common Patterns:**
{error_patterns}""",
            
            max_context_length=3500,
            required_context_types=['code', 'error', 'documentation']
        )
        
        # Documentation Template
        self.templates[PromptType.DOCUMENTATION] = PromptTemplate(
            name="documentation",
            prompt_type=PromptType.DOCUMENTATION,
            system_prompt="""You are a technical writer specializing in software documentation with expertise in creating clear, comprehensive, and user-friendly documentation.

Your skills include:
- API documentation and OpenAPI specifications
- User guides and tutorials
- Code documentation and inline comments
- Architecture documentation
- Setup and deployment guides
- Troubleshooting guides""",
            
            user_template="""Create documentation for:

**Subject:** {subject}
**Documentation Type:** {doc_type}
**Target Audience:** {audience}

{context_section}

Requirements:
- Clear and concise language
- Practical examples
- Step-by-step instructions where applicable
- Proper formatting and structure

Documentation:""",
            
            context_template="""**Reference Material:**

{context_items}

**Documentation Examples:**
{doc_examples}""",
            
            max_context_length=4000,
            required_context_types=['documentation', 'code']
        )
    
    def _initialize_context_strategies(self):
        """Initialize context injection strategies"""
        
        self.context_strategies = {
            'hierarchical': self._hierarchical_context_strategy,
            'similarity_based': self._similarity_based_strategy,
            'type_prioritized': self._type_prioritized_strategy,
            'balanced': self._balanced_strategy
        }
    
    async def build_prompt(self, 
                          task_description: str,
                          prompt_type: PromptType,
                          context_results: List[ContextResult],
                          additional_params: Dict[str, Any] = None,
                          strategy: str = 'balanced') -> Dict[str, str]:
        """
        Build a context-aware prompt
        
        Args:
            task_description: Main task description
            prompt_type: Type of prompt to generate
            context_results: Retrieved context results
            additional_params: Additional parameters for template
            strategy: Context injection strategy
            
        Returns:
            Dictionary with system_prompt and user_prompt
        """
        
        try:
            # Get appropriate template
            template = self.templates.get(prompt_type)
            if not template:
                logger.warning(f"No template found for {prompt_type}, using general template")
                template = self._get_general_template()
            
            # Apply context strategy
            context_strategy = self.context_strategies.get(strategy, self._balanced_strategy)
            processed_context = await context_strategy(context_results, template)
            
            # Build context section
            context_section = self._build_context_section(processed_context, template)
            
            # Prepare template parameters
            params = {
                'task_description': task_description,
                'context_section': context_section,
                'timestamp': datetime.now().isoformat()
            }
            
            # Add additional parameters
            if additional_params:
                params.update(additional_params)
            
            # Fill template
            user_prompt = template.user_template.format(**params)
            
            # Record prompt generation
            self._record_prompt_generation(prompt_type, strategy, len(context_results))
            
            return {
                'system_prompt': template.system_prompt,
                'user_prompt': user_prompt,
                'template_name': template.name,
                'context_count': len(context_results),
                'strategy_used': strategy
            }
            
        except Exception as e:
            logger.error(f"Error building prompt: {str(e)}")
            return self._get_fallback_prompt(task_description)
    
    async def _hierarchical_context_strategy(self, context_results: List[ContextResult],
                                           template: PromptTemplate) -> Dict[str, Any]:
        """Hierarchical context organization strategy"""
        
        # Group by context type and relevance
        context_groups = {}
        for result in context_results:
            context_type = result.context_type
            if context_type not in context_groups:
                context_groups[context_type] = []
            context_groups[context_type].append(result)
        
        # Sort each group by relevance
        for context_type in context_groups:
            context_groups[context_type].sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Prioritize required context types
        prioritized_context = {}
        if template.required_context_types:
            for req_type in template.required_context_types:
                if req_type in context_groups:
                    prioritized_context[req_type] = context_groups[req_type]
        
        # Add remaining context types
        for context_type, results in context_groups.items():
            if context_type not in prioritized_context:
                prioritized_context[context_type] = results
        
        return {
            'grouped_context': prioritized_context,
            'total_length': sum(len(r.content) for results in prioritized_context.values() for r in results)
        }
    
    async def _similarity_based_strategy(self, context_results: List[ContextResult],
                                       template: PromptTemplate) -> Dict[str, Any]:
        """Similarity-based context selection strategy"""
        
        # Sort by relevance score
        sorted_results = sorted(context_results, key=lambda x: x.relevance_score, reverse=True)
        
        # Select top results within length limit
        selected_results = []
        total_length = 0
        
        for result in sorted_results:
            if total_length + len(result.content) <= template.max_context_length:
                selected_results.append(result)
                total_length += len(result.content)
            else:
                break
        
        return {
            'selected_context': selected_results,
            'total_length': total_length
        }
    
    async def _type_prioritized_strategy(self, context_results: List[ContextResult],
                                       template: PromptTemplate) -> Dict[str, Any]:
        """Type-prioritized context selection strategy"""
        
        # Define type priorities
        type_priorities = {
            'code': 1.0,
            'api': 0.9,
            'documentation': 0.8,
            'example': 0.7,
            'tutorial': 0.6,
            'config': 0.5,
            'error': 0.4
        }
        
        # Calculate weighted scores
        for result in context_results:
            type_priority = type_priorities.get(result.context_type, 0.3)
            result.weighted_score = result.relevance_score * type_priority
        
        # Sort by weighted score
        sorted_results = sorted(context_results, key=lambda x: x.weighted_score, reverse=True)
        
        # Select within length limit
        selected_results = []
        total_length = 0
        
        for result in sorted_results:
            if total_length + len(result.content) <= template.max_context_length:
                selected_results.append(result)
                total_length += len(result.content)
        
        return {
            'selected_context': selected_results,
            'total_length': total_length
        }
    
    async def _balanced_strategy(self, context_results: List[ContextResult],
                               template: PromptTemplate) -> Dict[str, Any]:
        """Balanced context selection combining multiple factors"""
        
        # Combine hierarchical and similarity approaches
        hierarchical_result = await self._hierarchical_context_strategy(context_results, template)
        
        # Ensure diversity in context types
        selected_results = []
        total_length = 0
        type_counts = {}
        
        # First pass: select top result from each type
        for context_type, results in hierarchical_result['grouped_context'].items():
            if results and total_length + len(results[0].content) <= template.max_context_length:
                selected_results.append(results[0])
                total_length += len(results[0].content)
                type_counts[context_type] = 1
        
        # Second pass: fill remaining space with high-relevance results
        all_remaining = []
        for context_type, results in hierarchical_result['grouped_context'].items():
            all_remaining.extend(results[type_counts.get(context_type, 0):])
        
        # Sort remaining by relevance
        all_remaining.sort(key=lambda x: x.relevance_score, reverse=True)
        
        for result in all_remaining:
            if total_length + len(result.content) <= template.max_context_length:
                selected_results.append(result)
                total_length += len(result.content)
        
        return {
            'selected_context': selected_results,
            'total_length': total_length,
            'type_distribution': type_counts
        }
    
    def _build_context_section(self, processed_context: Dict[str, Any],
                             template: PromptTemplate) -> str:
        """Build the context section of the prompt"""
        
        if 'selected_context' in processed_context:
            # Simple list format
            context_items = []
            for i, result in enumerate(processed_context['selected_context'], 1):
                context_item = f"""**Context {i}** (from {result.source}, relevance: {result.relevance_score:.2f}):
```
{result.content}
```"""
                context_items.append(context_item)
            
            context_text = '\n\n'.join(context_items)
            
        elif 'grouped_context' in processed_context:
            # Grouped format
            context_sections = []
            for context_type, results in processed_context['grouped_context'].items():
                if results:
                    section_items = []
                    for result in results[:3]:  # Limit per type
                        section_items.append(f"- {result.content[:200]}..." if len(result.content) > 200 else f"- {result.content}")
                    
                    section = f"**{context_type.upper()} Context:**\n" + '\n'.join(section_items)
                    context_sections.append(section)
            
            context_text = '\n\n'.join(context_sections)
        
        else:
            context_text = "No relevant context found."
        
        # Apply context template if available
        if template.context_template and context_text:
            try:
                return template.context_template.format(
                    context_items=context_text,
                    patterns="",  # Could be enhanced with pattern matching
                    auth_patterns="",  # API-specific patterns
                    db_patterns="",  # Database patterns
                    error_patterns="",  # Error patterns
                    doc_examples=""  # Documentation examples
                )
            except KeyError:
                # Fallback if template has missing placeholders
                return context_text
        
        return context_text
    
    def _get_general_template(self) -> PromptTemplate:
        """Get a general-purpose template"""
        
        return PromptTemplate(
            name="general",
            prompt_type=PromptType.GENERAL,
            system_prompt="""You are an expert software developer and technical consultant with broad knowledge across multiple domains.

Provide accurate, helpful, and actionable responses based on the context and requirements provided.""",
            
            user_template="""Task: {task_description}

{context_section}

Please provide a comprehensive response addressing the task requirements.

Response:""",
            
            context_template="""**Relevant Context:**

{context_items}""",
            
            max_context_length=3000
        )
    
    def _get_fallback_prompt(self, task_description: str) -> Dict[str, str]:
        """Get fallback prompt when template processing fails"""
        
        return {
            'system_prompt': "You are a helpful AI assistant with expertise in software development.",
            'user_prompt': f"Please help with the following task:\n\n{task_description}",
            'template_name': 'fallback',
            'context_count': 0,
            'strategy_used': 'none'
        }
    
    def _record_prompt_generation(self, prompt_type: PromptType, strategy: str, context_count: int):
        """Record prompt generation for analytics"""
        
        record = {
            'timestamp': datetime.now().isoformat(),
            'prompt_type': prompt_type.value,
            'strategy': strategy,
            'context_count': context_count
        }
        
        self.prompt_history.append(record)
        
        # Update template usage statistics
        if prompt_type in self.templates:
            self.templates[prompt_type].usage_count += 1
    
    def record_prompt_success(self, prompt_type: PromptType, success: bool):
        """Record prompt success for template optimization"""
        
        if prompt_type in self.templates:
            template = self.templates[prompt_type]
            
            # Update success rate using exponential moving average
            alpha = 0.1  # Learning rate
            if template.usage_count == 1:
                template.success_rate = 1.0 if success else 0.0
            else:
                current_success = 1.0 if success else 0.0
                template.success_rate = (1 - alpha) * template.success_rate + alpha * current_success
    
    def get_template_stats(self) -> Dict[str, Any]:
        """Get statistics about template usage and success rates"""
        
        stats = {}
        for prompt_type, template in self.templates.items():
            stats[prompt_type.value] = {
                'usage_count': template.usage_count,
                'success_rate': template.success_rate,
                'name': template.name
            }
        
        return {
            'template_stats': stats,
            'total_prompts_generated': len(self.prompt_history),
            'recent_activity': self.prompt_history[-10:] if self.prompt_history else []
        }

class AdaptivePromptBuilder(ContextAwarePromptBuilder):
    """Enhanced prompt builder with adaptive learning capabilities"""
    
    def __init__(self):
        super().__init__()
        self.user_preferences = {}
        self.domain_adaptations = {}
    
    async def build_adaptive_prompt(self,
                                  task_description: str,
                                  prompt_type: PromptType,
                                  context_results: List[ContextResult],
                                  user_id: str = None,
                                  domain: str = None,
                                  **kwargs) -> Dict[str, str]:
        """Build prompt with user and domain adaptations"""
        
        # Apply user preferences
        if user_id and user_id in self.user_preferences:
            kwargs.update(self.user_preferences[user_id])
        
        # Apply domain adaptations
        if domain and domain in self.domain_adaptations:
            # Modify context strategy based on domain
            domain_config = self.domain_adaptations[domain]
            kwargs['strategy'] = domain_config.get('preferred_strategy', 'balanced')
        
        return await self.build_prompt(task_description, prompt_type, context_results, kwargs)
    
    def learn_user_preferences(self, user_id: str, feedback: Dict[str, Any]):
        """Learn user preferences from feedback"""
        
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {}
        
        # Update preferences based on feedback
        prefs = self.user_preferences[user_id]
        
        if 'preferred_detail_level' in feedback:
            prefs['detail_level'] = feedback['preferred_detail_level']
        
        if 'preferred_code_style' in feedback:
            prefs['code_style'] = feedback['preferred_code_style']
        
        if 'context_preference' in feedback:
            prefs['context_strategy'] = feedback['context_preference']
    
    def adapt_to_domain(self, domain: str, adaptations: Dict[str, Any]):
        """Add domain-specific adaptations"""
        
        self.domain_adaptations[domain] = adaptations
