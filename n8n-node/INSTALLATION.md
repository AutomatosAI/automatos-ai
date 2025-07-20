
# Installation Guide: n8n Multi-Agent Orchestrator Node

This guide provides detailed instructions for installing and configuring the Multi-Agent Orchestrator node in various n8n environments.

## Prerequisites

- n8n version 0.128.0 or higher
- Node.js 18+ (for development)
- Self-hosted n8n instance (community nodes not supported in n8n Cloud)
- Multi-Agent Orchestrator System running and accessible

## Installation Methods

### Method 1: Community Nodes UI (Recommended)

This is the easiest method for most users:

1. **Access n8n Settings**
   - Open your n8n instance
   - Navigate to **Settings** → **Community Nodes**

2. **Install the Node**
   - Click **"Install a community node"**
   - Enter: `n8n-nodes-multi-agent-orchestrator`
   - Click **"Install"**

3. **Restart n8n**
   - The node will be available after n8n restarts
   - Look for "Multi-Agent Orchestrator" in the node palette

### Method 2: npm Installation

For manual installation via npm:

```bash
# Navigate to your n8n installation directory
cd ~/.n8n/

# Install the package
npm install n8n-nodes-multi-agent-orchestrator

# Restart n8n
n8n start
```

### Method 3: Docker Installation

#### Option A: Dockerfile
Add to your n8n Dockerfile:

```dockerfile
FROM n8nio/n8n:latest

USER root

# Install the community node
RUN cd /usr/local/lib/node_modules/n8n && \
    npm install n8n-nodes-multi-agent-orchestrator

USER node
```

#### Option B: Docker Compose
Update your `docker compose.yml`:

```yaml
version: '3.8'
services:
  n8n:
    image: n8nio/n8n:latest
    environment:
      - N8N_COMMUNITY_PACKAGES_ENABLED=true
    volumes:
      - n8n_data:/home/node/.n8n
    ports:
      - "5678:5678"
    command: >
      sh -c "
        npm install n8n-nodes-multi-agent-orchestrator &&
        n8n start
      "
```

#### Option C: Init Container
Use an init container to install the node:

```yaml
version: '3.8'
services:
  n8n-init:
    image: n8nio/n8n:latest
    volumes:
      - n8n_data:/home/node/.n8n
    command: npm install n8n-nodes-multi-agent-orchestrator
    
  n8n:
    image: n8nio/n8n:latest
    depends_on:
      - n8n-init
    environment:
      - N8N_COMMUNITY_PACKAGES_ENABLED=true
    volumes:
      - n8n_data:/home/node/.n8n
    ports:
      - "5678:5678"
```

### Method 4: Development Installation

For developers or custom builds:

```bash
# Clone the repository
git clone https://github.com/your-org/n8n-nodes-multi-agent-orchestrator
cd n8n-nodes-multi-agent-orchestrator

# Install dependencies
npm install

# Build the node
npm run build

# Link to your n8n installation
npm link

# In your n8n directory
cd ~/.n8n/
npm link n8n-nodes-multi-agent-orchestrator

# Restart n8n
n8n start
```

## Configuration

### 1. Create API Credentials

After installation, you need to configure credentials:

1. **Navigate to Credentials**
   - Go to **Credentials** in your n8n instance
   - Click **"Add Credential"**

2. **Select Credential Type**
   - Search for "Multi-Agent Orchestrator API"
   - Click to create new credential

3. **Configure Settings**
   ```
   API Base URL: https://your-orchestrator.example.com
   API Key: mao_your_api_key_here
   Timeout: 300 (seconds)
   ```

4. **Test Connection**
   - Click **"Test"** to verify connectivity
   - Should return: `{"status": "healthy"}`

### 2. Environment Variables

For Docker deployments, set these environment variables:

```bash
# Enable community packages
N8N_COMMUNITY_PACKAGES_ENABLED=true

# Optional: Set custom extensions path
N8N_CUSTOM_EXTENSIONS=/home/node/.n8n/custom

# Optional: Allow external packages in Function nodes
NODE_FUNCTION_ALLOW_EXTERNAL=*
```

### 3. Network Configuration

Ensure your n8n instance can reach your Multi-Agent Orchestrator:

```bash
# Test connectivity
curl -H "Authorization: Bearer your_api_key" \
     https://your-orchestrator.example.com/api/v1/health
```

## Verification

### 1. Check Node Availability
- Open n8n workflow editor
- Click **"+"** to add a node
- Search for "Multi-Agent Orchestrator"
- Node should appear in search results

### 2. Test Basic Operation
Create a simple workflow:

1. **Add Manual Trigger**
2. **Add Multi-Agent Orchestrator Node**
3. **Configure**:
   - Resource: Agent
   - Operation: List
   - Credential: Your configured credential
4. **Execute Workflow**
5. **Verify Results**: Should return list of available agents

### 3. Check Logs
Monitor n8n logs for any errors:

```bash
# For standard installation
tail -f ~/.n8n/logs/n8n.log

# For Docker
docker logs -f n8n_container_name
```

## Troubleshooting

### Common Issues

#### 1. Node Not Appearing
**Problem**: Node doesn't show up in the palette

**Solutions**:
```bash
# Check if package is installed
npm list n8n-nodes-multi-agent-orchestrator

# Verify community packages are enabled
echo $N8N_COMMUNITY_PACKAGES_ENABLED

# Restart n8n completely
pkill -f n8n
n8n start
```

#### 2. Authentication Errors
**Problem**: 401 Unauthorized errors

**Solutions**:
- Verify API key format: `mao_...`
- Check base URL (no trailing slash)
- Test API key with curl:
```bash
curl -H "Authorization: Bearer your_api_key" \
     https://your-orchestrator.example.com/api/v1/health
```

#### 3. Connection Timeouts
**Problem**: Requests timing out

**Solutions**:
- Increase timeout in credential settings
- Check network connectivity
- Verify orchestrator system is running
- Check firewall rules

#### 4. Docker Issues
**Problem**: Node not loading in Docker

**Solutions**:
```bash
# Check if package is installed in container
docker exec n8n_container npm list n8n-nodes-multi-agent-orchestrator

# Verify environment variables
docker exec n8n_container env | grep N8N_COMMUNITY

# Rebuild container if needed
docker compose down
docker compose build --no-cache
docker compose up
```

#### 5. Version Compatibility
**Problem**: Node not compatible with n8n version

**Solutions**:
- Check n8n version: `n8n --version`
- Verify minimum version requirement (0.128.0+)
- Update n8n if needed:
```bash
npm update -g n8n
```

### Debug Mode

Enable debug logging for detailed troubleshooting:

```bash
# Set debug environment
export N8N_LOG_LEVEL=debug
export DEBUG=n8n*

# Start n8n
n8n start
```

### Getting Help

If you encounter issues:

1. **Check Documentation**: [Multi-Agent Orchestrator Docs](https://docs.yourorg.com)
2. **GitHub Issues**: [Report bugs](https://github.com/your-org/n8n-nodes-multi-agent-orchestrator/issues)
3. **n8n Community**: [Community Forum](https://community.n8n.io/)
4. **Support Email**: support@yourorg.com

## Security Considerations

### API Key Management
- Store API keys securely in n8n credentials
- Use environment variables for sensitive data
- Rotate API keys regularly
- Limit API key permissions to minimum required

### Network Security
- Use HTTPS for all API communications
- Implement proper firewall rules
- Consider VPN for internal deployments
- Monitor API access logs

### Docker Security
```dockerfile
# Use non-root user
USER node

# Limit container capabilities
--cap-drop=ALL

# Use read-only filesystem where possible
--read-only

# Set resource limits
--memory=1g --cpus=1
```

## Performance Optimization

### Timeout Settings
- Set appropriate timeouts based on operation type
- Use async operations for long-running tasks
- Implement proper retry logic

### Batch Operations
- Process multiple items efficiently
- Use pagination for large datasets
- Implement proper error handling

### Monitoring
- Monitor API response times
- Track error rates
- Set up alerts for failures

## Next Steps

After successful installation:

1. **Explore Examples**: Check the README for usage examples
2. **Build Workflows**: Create your first multi-agent workflow
3. **Integration**: Connect with existing n8n workflows
4. **Monitoring**: Set up logging and monitoring
5. **Scaling**: Plan for production deployment

---

For additional support, please refer to the main documentation or contact our support team.
