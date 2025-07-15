
# n8n-nodes-multi-agent-orchestrator

This is an n8n community node that integrates with Multi-Agent Orchestration Systems, enabling seamless workflow automation, document management, and context engineering within n8n workflows.

![n8n.io - Workflow Automation](https://raw.githubusercontent.com/n8n-io/n8n/master/assets/n8n-logo.png)

## Features

### 🔄 Workflow Operations
- **Start Workflows**: Trigger multi-agent workflows with custom parameters
- **Monitor Status**: Track workflow execution progress and results
- **Stop Workflows**: Gracefully terminate running workflows
- **List Executions**: View all workflow executions and their status

### 📄 Document Management
- **Upload Documents**: Add PDF, Word, text, and other document types
- **Retrieve Documents**: Access document metadata and content
- **Delete Documents**: Remove documents from the system
- **List Documents**: Browse all available documents with filtering

### 🧠 Context Engineering
- **Semantic Search**: Find relevant information using natural language queries
- **Add Context**: Insert new knowledge, procedures, templates, and references
- **Update Context**: Modify existing context entries
- **Delete Context**: Remove outdated or incorrect context information

### 🤖 Agent Management
- **List Agents**: View all available agents and their capabilities
- **Agent Status**: Check agent health and current workload
- **Execute Tasks**: Run specific tasks using specialized agents

## Installation

To install this node in your n8n instance:

### Option 1: Community Nodes (Recommended)
1. Go to **Settings** > **Community Nodes** in your n8n instance
2. Click **Install a community node**
3. Enter `n8n-nodes-multi-agent-orchestrator`
4. Click **Install**

### Option 2: Manual Installation
```bash
# Navigate to your n8n installation directory
cd ~/.n8n/

# Install the node package
npm install n8n-nodes-multi-agent-orchestrator

# Restart n8n
n8n start
```

### Option 3: Docker Installation
Add the following to your n8n Docker configuration:

```dockerfile
# In your Dockerfile
RUN cd /usr/local/lib/node_modules/n8n && npm install n8n-nodes-multi-agent-orchestrator

# Or using environment variable
ENV N8N_COMMUNITY_PACKAGES_ENABLED=true
```

## Configuration

### 1. Set up Credentials
1. In n8n, go to **Credentials** and create a new credential
2. Search for "Multi-Agent Orchestrator API"
3. Configure the following:
   - **API Base URL**: Your orchestrator system URL (e.g., `https://orchestrator.yourcompany.com`)
   - **API Key**: Your authentication token
   - **Timeout**: Request timeout in seconds (default: 300)

### 2. Test Connection
Click **Test** to verify your credentials are working correctly.

## Usage Examples

### Example 1: Start a Data Analysis Workflow
```json
{
  "resource": "workflow",
  "operation": "start",
  "workflowTemplate": "data-analysis-pipeline",
  "inputParameters": {
    "dataset_url": "https://example.com/data.csv",
    "analysis_type": "trend_analysis",
    "output_format": "report"
  },
  "priority": "high"
}
```

### Example 2: Upload and Process Document
```json
{
  "resource": "document",
  "operation": "upload",
  "fileName": "quarterly-report.pdf",
  "fileType": "pdf",
  "fileContent": "base64-encoded-content",
  "tags": "quarterly, financial, 2024"
}
```

### Example 3: Search Context for Information
```json
{
  "resource": "context",
  "operation": "search",
  "searchQuery": "What are the best practices for customer onboarding?",
  "maxResults": 5
}
```

### Example 4: Execute Agent Task
```json
{
  "resource": "agent",
  "operation": "executeTask",
  "agentId": "data-analyst-agent",
  "taskDescription": "Analyze sales trends from the uploaded dataset",
  "taskParameters": {
    "time_period": "last_quarter",
    "metrics": ["revenue", "conversion_rate", "customer_acquisition"]
  }
}
```

## API Endpoints

The node interacts with the following API endpoints:

### Workflows
- `POST /api/v1/workflows` - Start workflow
- `GET /api/v1/workflows/{id}` - Get workflow status
- `POST /api/v1/workflows/{id}/stop` - Stop workflow
- `GET /api/v1/workflows` - List workflows

### Documents
- `POST /api/v1/documents` - Upload document
- `GET /api/v1/documents/{id}` - Get document
- `DELETE /api/v1/documents/{id}` - Delete document
- `GET /api/v1/documents` - List documents

### Context
- `POST /api/v1/context/search` - Search context
- `POST /api/v1/context` - Add context
- `PUT /api/v1/context/{id}` - Update context
- `DELETE /api/v1/context/{id}` - Delete context

### Agents
- `GET /api/v1/agents` - List agents
- `GET /api/v1/agents/{id}` - Get agent status
- `POST /api/v1/agents/{id}/execute` - Execute agent task

## Error Handling

The node includes comprehensive error handling:

- **401 Unauthorized**: Invalid API credentials
- **404 Not Found**: Resource doesn't exist
- **500+ Server Errors**: System issues
- **Timeout Errors**: Request exceeded timeout limit

Enable "Continue on Fail" in your workflow to handle errors gracefully.

## Advanced Features

### Async Operations
Enable asynchronous execution for long-running operations:
```json
{
  "additionalOptions": {
    "async": true,
    "callbackUrl": "https://your-webhook.example.com/callback"
  }
}
```

### Custom Timeouts
Adjust timeout for specific operations:
```json
{
  "additionalOptions": {
    "timeout": 600
  }
}
```

## Troubleshooting

### Common Issues

1. **Node not appearing**: Ensure n8n community packages are enabled
2. **Authentication errors**: Verify API key and base URL
3. **Timeout issues**: Increase timeout in additional options
4. **Connection refused**: Check if orchestrator system is accessible

### Debug Mode
Enable debug logging in your n8n instance:
```bash
export N8N_LOG_LEVEL=debug
n8n start
```

## Development

### Building from Source
```bash
git clone https://github.com/your-org/n8n-nodes-multi-agent-orchestrator
cd n8n-nodes-multi-agent-orchestrator
npm install
npm run build
```

### Testing
```bash
npm run test
npm run lint
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Support

- **Documentation**: [https://docs.yourorg.com/multi-agent-orchestrator](https://docs.yourorg.com/multi-agent-orchestrator)
- **Issues**: [GitHub Issues](https://github.com/your-org/n8n-nodes-multi-agent-orchestrator/issues)
- **Community**: [n8n Community Forum](https://community.n8n.io/)

## License

[MIT](LICENSE.md)

## Changelog

### v1.0.0
- Initial release
- Support for workflow, document, context, and agent operations
- Comprehensive error handling
- Async operation support
- Full TypeScript implementation

---

**Note**: This node requires a self-hosted n8n instance as community nodes are not supported in n8n Cloud.
