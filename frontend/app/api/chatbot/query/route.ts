import { NextRequest, NextResponse } from 'next/server'
import Anthropic from '@anthropic-ai/sdk'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'https://api.automatos.app'
const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY

const anthropic = new Anthropic({
  apiKey: ANTHROPIC_API_KEY,
})

interface ChatContext {
  currentPage: string
  selectedItems: any[]
  userRole: string
  recentActions: any[]
}

// Smart intent detection
function detectIntent(query: string): {
  intent: string
  confidence: number
  source: 'codegraph' | 'rag' | 'semantic' | 'llm'
} {
  const q = query.toLowerCase()
  
  console.log(`[Intent] Analyzing query: "${query}"`)
  
  // Document patterns (RAG) - CHECK FIRST!
  const docPatterns = [
    /\b(document|file|pdf|upload|search documents|find in documents)\b/i,
    /\b(documentation|guide|tutorial|readme|manual|docs)\b/i,
    /^(show|find|search|get).*\b(document|documentation|guide|manual|readme|setup)\b/i,
    /^(what|where|how).*\b(documentation|docs|guide|manual|setup|deploy|install|configure)\b/i,
    /\b(deployment|setup|installation|configuration|infrastructure|architecture)\b/i, // Match these words directly
    /\b(deploy|install|configure|readme|prd|spec|specification|requirement)\b/i, // Common doc-related terms
    /^(tell me about|explain|describe|what is|how to).*\b(deploy|setup|architecture|design|system|platform)\b/i // Explanatory questions
  ]
  
  for (const pattern of docPatterns) {
    if (pattern.test(query)) {
      console.log(`[Intent] Matched document pattern: ${pattern}`)
      return { intent: 'document_search', confidence: 0.85, source: 'rag' }
    }
  }
  
  // CodeGraph patterns (code-related questions) - CHECK AFTER DOCUMENTS!
  const codePatterns = [
    /\b(function|class|method|variable|code|implementation)\b/i,
    /\b(show|find|get|explain|where).*\b(code|function|class|method|factory|service|component|api|endpoint)\b/i,
    /^(how|what|where|why|explain).*\b(work|works|authenticate|authorization|api|endpoint|route|handler|factory|agent|workflow|execute)\b/i,
    /\b(github|repository|codebase|source|file|line|symbol|import|export)\b/i,
    /\b(agent|workflow|factory|manager|handler|controller|service).*\b(code|implementation|work|works)\b/i,
    /\bagentfactory\b/i, // Match AgentFactory directly
    /\b(agent|workflow|execution|orchestrat|authentication|authorization)\w*\b/i // Match common terms
  ]
  
  for (const pattern of codePatterns) {
    if (pattern.test(query)) {
      console.log(`[Intent] Matched code pattern: ${pattern}`)
      return { intent: 'code_search', confidence: 0.9, source: 'codegraph' }
    }
  }
  
  // System/Agent patterns (more specific to avoid false matches)
  const systemPatterns = [
    /\b(system health|performance metrics|system status)\b/i,
    /^(create|manage|monitor|optimize)\s+(a|an|the)?\s*(workflow|agent|system)/i
  ]
  
  for (const pattern of systemPatterns) {
    if (pattern.test(query)) {
      console.log(`[Intent] Matched system pattern: ${pattern}`)
      return { intent: 'system_query', confidence: 0.75, source: 'llm' }
    }
  }
  
  // Default to LLM
  console.log(`[Intent] No pattern matched, defaulting to LLM`)
  return { intent: 'general', confidence: 0.6, source: 'llm' }
}

// Extract key terms from query for better search
function extractSearchTerms(query: string): string {
  // Remove common conversational words
  const stopWords = ['show', 'me', 'the', 'find', 'get', 'explain', 'how', 'does', 'work', 'where', 'is', 'what', 'code', 'please', 'can', 'you']
  
  const terms = query
    .toLowerCase()
    .split(/\s+/)
    .filter(word => !stopWords.includes(word) && word.length > 2)
    .join(' ')
  
  console.log(`[CodeGraph] Extracted terms: "${terms}" from "${query}"`)
  return terms || query // fallback to original if nothing left
}

// CodeGraph search
async function searchCodeGraph(query: string): Promise<any> {
  try {
    // Extract meaningful search terms
    const searchTerms = extractSearchTerms(query)
    console.log(`[CodeGraph] Searching for: "${searchTerms}"`)
    
    // Try semantic search first
    const semanticResponse = await fetch(
      `${API_BASE_URL}/api/code-graph/search/semantic?project=Automatos-ai&q=${encodeURIComponent(searchTerms)}&limit=5`,
      { 
        headers: { 'Accept': 'application/json' },
        cache: 'no-store' // Prevent caching
      }
    )
    
    if (semanticResponse.ok) {
      const data = await semanticResponse.json()
      console.log(`[CodeGraph] Received ${data.count} results`)
      console.log(`[CodeGraph] Results:`, data.results.map((r: any) => `${r.name} (${r.symbol_type}) in ${r.file_path}`))
      
      if (data.count > 0) {
        const mapped = data.results.map((r: any) => ({
          language: r.file_path.endsWith('.py') ? 'python' : 
                   r.file_path.endsWith('.ts') || r.file_path.endsWith('.tsx') ? 'typescript' : 'javascript',
          code: r.code_snippet || r.signature,
          file_path: r.file_path,
          line_number: r.line_number,
          symbol_name: r.name,
          explanation: r.docstring || `${r.symbol_type}: ${r.name}`
        }))
        console.log(`[CodeGraph] Mapped results:`, mapped.map(m => `${m.symbol_name} in ${m.file_path}`))
          return {
          results: mapped,
          count: data.count,
          execution_time: data.execution_time_ms
        }
      }
    }
    
    // Fallback to symbol search
    const symbolResponse = await fetch(
      `${API_BASE_URL}/api/code-graph/search/symbols?project=Automatos-ai&q=${encodeURIComponent(searchTerms)}&limit=5`,
      { headers: { 'Accept': 'application/json' } }
    )
    
    if (symbolResponse.ok) {
      const data = await symbolResponse.json()
          return {
        results: data.results.map((r: any) => ({
          language: r.file_path.endsWith('.py') ? 'python' : 
                   r.file_path.endsWith('.ts') || r.file_path.endsWith('.tsx') ? 'typescript' : 'javascript',
          code: r.code_snippet || r.signature,
          file_path: r.file_path,
          line_number: r.line_number,
          symbol_name: r.name,
          explanation: r.docstring || `${r.symbol_type}: ${r.name}`
        })),
        count: data.count,
        execution_time: data.execution_time_ms
      }
    }
    
    return { results: [], count: 0, execution_time: 0 }
  } catch (error) {
    console.error('CodeGraph search error:', error)
    return { results: [], count: 0, execution_time: 0 }
  }
}

// Semantic search for documents (similar to CodeGraph)
async function searchDocuments(query: string): Promise<any> {
  try {
    // Use same term extraction for consistency
    const searchTerms = extractSearchTerms(query)
    console.log(`[Documents] Searching for: "${searchTerms}"`)
    
    // Try semantic search first (pgvector, just like CodeGraph)
    // Note: Backend uses POST with query params (FastAPI Query parameters)
    const semanticResponse = await fetch(
      `${API_BASE_URL}/api/documents/search?query=${encodeURIComponent(searchTerms)}&limit=8&min_similarity=0.65`,
      { 
        method: 'POST',  // Backend expects POST!
        headers: { 'Accept': 'application/json' },
        cache: 'no-store'
      }
    )
    
    if (semanticResponse.ok) {
      const data = await semanticResponse.json()
      console.log(`[Documents] Semantic search received ${data.total_results || 0} results`)
      
      if (data.results && data.results.length > 0) {
        console.log(`[Documents] Results:`, data.results.map((r: any) => 
          `${r.source?.filename || 'Unknown'} (${(r.similarity * 100).toFixed(1)}%)`
        ))
        
          return {
          chunks: data.results,
          execution_time: data.execution_time_ms
        }
      }
    }
    
    // Fallback to RAG retrieve if semantic search fails or returns nothing
    console.log(`[Documents] Trying RAG retrieve as fallback...`)
    const ragResponse = await fetch(`${API_BASE_URL}/api/documents/rag/retrieve`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: searchTerms,
        max_chunks: 5,
        max_tokens: 2000,
        diversity: 0.3
      })
    })
    
    if (ragResponse.ok) {
      const data = await ragResponse.json()
      console.log(`[Documents] RAG received ${data.chunks?.length || 0} chunks`)
      
      if (data.chunks && data.chunks.length > 0) {
        return {
          chunks: data.chunks,
          context: data.context || '',
          execution_time: data.execution_time_ms
        }
      }
    }
    
    // Last resort: List available documents by filename match
    console.log(`[Documents] Trying filename match as last resort...`)
    const listResponse = await fetch(`${API_BASE_URL}/api/documents/`, {
      headers: { 'Accept': 'application/json' },
      cache: 'no-store'
    })
    
    if (listResponse.ok) {
      const allDocs = await listResponse.json()
      
      // Filter documents by filename match
      const searchLower = searchTerms.toLowerCase()
      const matchingDocs = allDocs.filter((doc: any) => {
        const filenameLower = doc.filename?.toLowerCase() || ''
        const descLower = doc.description?.toLowerCase() || ''
        return filenameLower.includes(searchLower) || 
               descLower.includes(searchLower) ||
               doc.tags?.some((tag: string) => tag.toLowerCase().includes(searchLower))
      }).slice(0, 5)
      
      if (matchingDocs.length > 0) {
        console.log(`[Documents] Found ${matchingDocs.length} matching files:`, 
          matchingDocs.map((d: any) => d.filename))
        
        // Convert to chunk format
          return {
          chunks: matchingDocs.map((doc: any) => ({
            id: doc.id,
            document_id: doc.id,
            filename: doc.filename,
            content: doc.description || `Document: ${doc.filename}\nFile type: ${doc.file_type}\nChunks: ${doc.chunk_count}\nUploaded: ${doc.upload_date}`,
            similarity: 0.85, // Filename match confidence
            source: {
              filename: doc.filename,
              file_type: doc.file_type,
              file_size: doc.file_size,
              chunk_count: doc.chunk_count
            }
          })),
          execution_time: 0
        }
      }
    }
    
    console.log(`[Documents] No results from any method`)
    return { chunks: [], context: '', execution_time: 0 }
  } catch (error) {
    console.error('[Documents] Search error:', error)
    return { chunks: [], context: '', execution_time: 0 }
  }
}

// Generate LLM response using Anthropic Claude
async function generateLLMResponse(
  query: string, 
  codeSnippets: any[], 
  documents: any[], 
  source: string
): Promise<string> {
  try {
    // Build context for Claude
    let contextPrompt = ''
    
    console.log(`[Claude] Building context - Source: ${source}, Code: ${codeSnippets.length}, Docs: ${documents.length}`)
    
    if (source === 'codegraph' && codeSnippets.length > 0) {
      contextPrompt = `\n\n<code_context>\nI found ${codeSnippets.length} relevant code symbols from the Automatos codebase:\n\n`
      codeSnippets.forEach((snippet, idx) => {
        contextPrompt += `${idx + 1}. **${snippet.symbol_name}** in \`${snippet.file_path}\` (Line ${snippet.line_number})\n`
        contextPrompt += `\`\`\`${snippet.language}\n${snippet.code}\n\`\`\`\n`
        if (snippet.explanation) {
          contextPrompt += `Documentation: ${snippet.explanation}\n`
        }
        contextPrompt += '\n'
      })
      contextPrompt += '</code_context>\n\n'
    } else if (source === 'rag' && documents.length > 0) {
      contextPrompt = `\n\n<document_context>\nI found ${documents.length} relevant documents from your knowledge base:\n\n`
      documents.forEach((doc, idx) => {
        contextPrompt += `## Document ${idx + 1}: ${doc.filename || 'Untitled'}\n`
        if (doc.similarity) {
          contextPrompt += `Match Confidence: ${(doc.similarity * 100).toFixed(1)}%\n`
        }
        contextPrompt += `\n${doc.excerpt}\n\n`
        contextPrompt += `---\n\n`
      })
      contextPrompt += '</document_context>\n\n'
    }
    
    // System prompt for Automatos AI Assistant
    const systemPrompt = `You are the Automatos AI Assistant, an expert coding assistant for the Automatos AI Platform.

CRITICAL INSTRUCTIONS:
- You HAVE DIRECT ACCESS to the user's codebase through CodeGraph semantic search
- You HAVE DIRECT ACCESS to uploaded documents through RAG retrieval
- When code snippets or documents are provided in the user message, they are REAL results from the actual codebase
- ALWAYS use the provided code/documents as your primary source of truth
- NEVER say you don't have access - the context provided IS your access

Your role:
1. Analyze the provided code snippets and documents carefully
2. Explain how the code works, referencing specific files and line numbers
3. Provide clear, technical explanations with code examples
4. Suggest improvements and best practices
5. Be direct and actionable

When responding:
- Start by directly addressing the question with the provided context
- Reference specific files, functions, and line numbers from the context
- Use the exact code snippets shown to explain concepts
- Format code with proper syntax highlighting
- If no context is provided, explain what you can do and suggest specific searches

Keep responses focused, technical, and practical.`

    const userPrompt = contextPrompt 
      ? `${contextPrompt}**User Question:** ${query}

The code/documents above are REAL RESULTS from the Automatos codebase. Use them to answer the question directly. Reference specific files, functions, and line numbers from the context provided.`
      : `**User Question:** ${query}

I searched the codebase but found no direct matches. Please provide a helpful general answer about Automatos and suggest what specific code/documentation I should search for.`

    console.log(`[Claude] Sending prompt with context length: ${contextPrompt.length}`)
    console.log(`[Claude] User prompt preview: ${userPrompt.slice(0, 200)}...`)

    // Call Anthropic Claude API
    const message = await anthropic.messages.create({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 2000,
      temperature: 0.7,
      system: systemPrompt,
      messages: [
        {
          role: 'user',
          content: userPrompt
        }
      ]
    })
    
    // Extract text response
    const response = message.content[0]
    const responseText = response.type === 'text' ? response.text : 'Unable to generate response.'
    
    console.log(`[Claude] Response length: ${responseText.length}`)
    console.log(`[Claude] Response preview: ${responseText.slice(0, 200)}...`)
    
    return responseText
    
  } catch (error: any) {
    console.error('Claude API error:', error)
    
    // Fallback to basic response on error
    if (source === 'codegraph' && codeSnippets.length > 0) {
      return `I found ${codeSnippets.length} code symbols related to "${query}".\n\nClick on any snippet below to view the full code.`
    }
    
    if (source === 'rag' && documents.length > 0) {
      return `I found ${documents.length} relevant documents about "${query}".\n\nView the document excerpts below for more details.`
    }
    
    return `I understand you're asking about "${query}". I'm currently experiencing connection issues with the AI model. Please try again in a moment.`
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { query, context, session_id } = body

    const startTime = Date.now()
    
    console.log(`[Chatbot] Query received: "${query}"`)
    
    // Use Claude to decide what to search with function calling
    const decisionMessage = await anthropic.messages.create({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      tools: [
        {
          name: 'search_code',
          description: 'Search the codebase for functions, classes, methods, and code implementations. Use this when the user asks about code, how something works, implementation details, or specific code symbols.',
          input_schema: {
            type: 'object',
            properties: {
              query: {
                type: 'string',
                description: 'The search query for code'
              }
            },
            required: ['query']
          }
        },
        {
          name: 'search_documents',
          description: 'Search documentation, guides, PDFs, and uploaded documents. Use this when the user asks about documentation, guides, architecture, deployment, setup, or general information.',
          input_schema: {
            type: 'object',
            properties: {
              query: {
                type: 'string',
                description: 'The search query for documents'
              }
            },
            required: ['query']
          }
        }
      ],
      messages: [
        {
          role: 'user',
          content: `User query: "${query}"\n\nDecide whether to search code or documents based on this query. If asking about code implementation or specific functions/classes, use search_code. If asking about documentation, guides, or general information, use search_documents.`
        }
      ]
    })
    
    let codeSnippets: any[] = []
    let documents: any[] = []
    let source = 'llm'
    
    // Execute the tool calls Claude decided on
    if (decisionMessage.content) {
      for (const block of decisionMessage.content) {
        if (block.type === 'tool_use') {
          console.log(`[Claude] Chose tool: ${block.name}`)
          if (block.name === 'search_code') {
            source = 'codegraph'
            const codeResults = await searchCodeGraph(query)
            codeSnippets = codeResults.results
          } else if (block.name === 'search_documents') {
            source = 'rag'
            const docResults = await searchDocuments(query)
            documents = docResults.chunks.map((chunk: any) => ({
              id: chunk.chunk_id || chunk.id,
              filename: chunk.source?.filename || chunk.filename || chunk.document_name || 'Document',
              excerpt: chunk.content || chunk.text || '',
              similarity: chunk.similarity || chunk.score || 0.8
            }))
          }
        }
      }
    }
    
    // Generate intelligent response using Claude with context
    const content = await generateLLMResponse(query, codeSnippets, documents, source)
    
    const processingTime = (Date.now() - startTime) / 1000
    
    return NextResponse.json({
      message: {
        id: `bot-${Date.now()}`,
        content,
        timestamp: new Date().toISOString(),
        code_snippets: codeSnippets,
        documents: documents,
        metadata: {
          source,
          processing_time: processingTime,
          llm_decision: true
        }
      },
      session_id
    })
    
  } catch (error: any) {
    console.error('Chatbot query error:', error)
    return NextResponse.json(
      { 
        error: 'Failed to process query',
        details: error.message 
      },
      { status: 500 }
    )
  }
}
