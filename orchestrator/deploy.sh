#!/bin/bash

# Enhanced Two-Tiered Multi-Agent Orchestration System
# Deployment Script for mcp.xplaincrypto.ai
# 
# This script deploys the complete orchestration system to the target server

set -e

# Configuration
DEPLOY_HOST="${DEPLOY_HOST:-mcp.xplaincrypto.ai}"
DEPLOY_USER="${DEPLOY_USER:-root}"
DEPLOY_PATH="/opt/enhanced_orchestrator_v2"
BACKUP_PATH="/backup/orchestrator_$(date +%Y%m%d_%H%M%S)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if we can connect to the server
    if ! ssh -o ConnectTimeout=10 "$DEPLOY_USER@$DEPLOY_HOST" "echo 'Connection test successful'" >/dev/null 2>&1; then
        error "Cannot connect to $DEPLOY_HOST. Please check SSH configuration."
        exit 1
    fi
    
    # Check if Docker is installed on target
    if ! ssh "$DEPLOY_USER@$DEPLOY_HOST" "command -v docker >/dev/null 2>&1"; then
        warning "Docker not found on target server. Will install Docker."
        INSTALL_DOCKER=true
    else
        info "Docker found on target server."
        INSTALL_DOCKER=false
    fi
    
    # Check if docker-compose is installed
    if ! ssh "$DEPLOY_USER@$DEPLOY_HOST" "command -v docker-compose >/dev/null 2>&1"; then
        warning "Docker Compose not found on target server. Will install Docker Compose."
        INSTALL_COMPOSE=true
    else
        info "Docker Compose found on target server."
        INSTALL_COMPOSE=false
    fi
}

# Install Docker on target server
install_docker() {
    if [ "$INSTALL_DOCKER" = true ]; then
        log "Installing Docker on target server..."
        ssh "$DEPLOY_USER@$DEPLOY_HOST" << 'EOF'
            # Update package index
            apt-get update
            
            # Install prerequisites
            apt-get install -y apt-transport-https ca-certificates curl gnupg lsb-release
            
            # Add Docker GPG key
            curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
            
            # Add Docker repository
            echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
            
            # Install Docker
            apt-get update
            apt-get install -y docker-ce docker-ce-cli containerd.io
            
            # Start and enable Docker
            systemctl start docker
            systemctl enable docker
            
            # Add user to docker group
            usermod -aG docker $USER
EOF
        log "Docker installation completed."
    fi
}

# Install Docker Compose on target server
install_docker_compose() {
    if [ "$INSTALL_COMPOSE" = true ]; then
        log "Installing Docker Compose on target server..."
        ssh "$DEPLOY_USER@$DEPLOY_HOST" << 'EOF'
            # Download Docker Compose
            curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
            
            # Make it executable
            chmod +x /usr/local/bin/docker-compose
            
            # Create symlink
            ln -sf /usr/local/bin/docker-compose /usr/bin/docker-compose
EOF
        log "Docker Compose installation completed."
    fi
}

# Create backup of existing deployment
create_backup() {
    log "Creating backup of existing deployment..."
    ssh "$DEPLOY_USER@$DEPLOY_HOST" << EOF
        if [ -d "$DEPLOY_PATH" ]; then
            mkdir -p "$BACKUP_PATH"
            cp -r "$DEPLOY_PATH" "$BACKUP_PATH/"
            echo "Backup created at $BACKUP_PATH"
        else
            echo "No existing deployment found, skipping backup."
        fi
EOF
}

# Deploy application files
deploy_files() {
    log "Deploying application files..."
    
    # Create deployment directory
    ssh "$DEPLOY_USER@$DEPLOY_HOST" "mkdir -p $DEPLOY_PATH"
    
    # Copy files to server
    rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
        ./ "$DEPLOY_USER@$DEPLOY_HOST:$DEPLOY_PATH/"
    
    log "Files deployed successfully."
}

# Configure environment
configure_environment() {
    log "Configuring environment..."
    
    ssh "$DEPLOY_USER@$DEPLOY_HOST" << EOF
        cd "$DEPLOY_PATH"
        
        # Create .env file if it doesn't exist
        if [ ! -f .env ]; then
            cp .env.example .env
            echo "Created .env file from template. Please configure it manually."
        fi
        
        # Create required directories
        mkdir -p logs vector_stores projects keys monitoring/grafana monitoring/prometheus
        
        # Set proper permissions
        chmod 755 logs vector_stores projects
        chmod 700 keys
        
        # Create SSH key if it doesn't exist
        if [ ! -f keys/deploy_key ]; then
            ssh-keygen -t ed25519 -f keys/deploy_key -N "" -C "orchestrator@$DEPLOY_HOST"
            echo "SSH key generated. Add the public key to authorized_keys:"
            cat keys/deploy_key.pub
        fi
        
        chmod 600 keys/deploy_key
        chmod 644 keys/deploy_key.pub
EOF
}

# Setup monitoring configuration
setup_monitoring() {
    log "Setting up monitoring configuration..."
    
    ssh "$DEPLOY_USER@$DEPLOY_HOST" << 'EOF'
        cd /opt/enhanced_orchestrator_v2
        
        # Create Prometheus configuration
        cat > monitoring/prometheus.yml << 'PROMETHEUS_EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'orchestrator'
    static_configs:
      - targets: ['mcp_bridge:8001']
    metrics_path: '/metrics'
    scrape_interval: 30s
PROMETHEUS_EOF

        # Create Grafana datasource configuration
        mkdir -p monitoring/grafana/datasources
        cat > monitoring/grafana/datasources/prometheus.yml << 'GRAFANA_EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
GRAFANA_EOF

        echo "Monitoring configuration created."
EOF
}

# Configure firewall
configure_firewall() {
    log "Configuring firewall..."
    
    ssh "$DEPLOY_USER@$DEPLOY_HOST" << 'EOF'
        # Install UFW if not present
        if ! command -v ufw >/dev/null 2>&1; then
            apt-get update
            apt-get install -y ufw
        fi
        
        # Reset firewall
        ufw --force reset
        
        # Default policies
        ufw default deny incoming
        ufw default allow outgoing
        
        # Allow SSH
        ufw allow 22/tcp
        
        # Allow HTTP/HTTPS
        ufw allow 80/tcp
        ufw allow 443/tcp
        
        # Allow orchestrator services
        ufw allow 8001/tcp  # MCP Bridge
        
        # Optional monitoring ports (comment out if not needed)
        # ufw allow 3000/tcp  # Grafana
        # ufw allow 9090/tcp  # Prometheus
        
        # Enable firewall
        ufw --force enable
        
        echo "Firewall configured successfully."
EOF
}

# Deploy services
deploy_services() {
    log "Deploying services..."
    
    ssh "$DEPLOY_USER@$DEPLOY_HOST" << EOF
        cd "$DEPLOY_PATH"
        
        # Stop existing services
        docker-compose down || true
        
        # Build and start services
        docker-compose up -d postgres redis
        
        # Wait for database to be ready
        echo "Waiting for database to be ready..."
        sleep 30
        
        # Start main services
        docker-compose up -d mcp_bridge
        
        # Check service health
        sleep 10
        docker-compose ps
EOF
}

# Verify deployment
verify_deployment() {
    log "Verifying deployment..."
    
    # Test API endpoint
    if ssh "$DEPLOY_USER@$DEPLOY_HOST" "curl -f http://localhost:8001/health >/dev/null 2>&1"; then
        log "✅ API health check passed"
    else
        error "❌ API health check failed"
        return 1
    fi
    
    # Check Docker containers
    ssh "$DEPLOY_USER@$DEPLOY_HOST" << EOF
        cd "$DEPLOY_PATH"
        echo "Container status:"
        docker-compose ps
        
        echo ""
        echo "Recent logs:"
        docker-compose logs --tail=20 mcp_bridge
EOF
}

# Setup SSL (optional)
setup_ssl() {
    if [ "$SETUP_SSL" = "true" ]; then
        log "Setting up SSL certificate..."
        
        ssh "$DEPLOY_USER@$DEPLOY_HOST" << 'EOF'
            # Install Nginx and Certbot
            apt-get update
            apt-get install -y nginx certbot python3-certbot-nginx
            
            # Create Nginx configuration
            cat > /etc/nginx/sites-available/orchestrator << 'NGINX_EOF'
server {
    listen 80;
    server_name mcp.xplaincrypto.ai;

    location / {
        proxy_pass http://localhost:8001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
NGINX_EOF
            
            # Enable site
            ln -sf /etc/nginx/sites-available/orchestrator /etc/nginx/sites-enabled/
            rm -f /etc/nginx/sites-enabled/default
            
            # Test configuration
            nginx -t
            
            # Restart Nginx
            systemctl restart nginx
            systemctl enable nginx
            
            # Obtain SSL certificate
            certbot --nginx -d mcp.xplaincrypto.ai --non-interactive --agree-tos --email admin@xplaincrypto.ai
            
            echo "SSL setup completed."
EOF
    fi
}

# Main deployment function
main() {
    log "Starting deployment of Enhanced Two-Tiered Multi-Agent Orchestration System"
    log "Target: $DEPLOY_USER@$DEPLOY_HOST"
    
    # Check if we should proceed
    if [ "$1" != "--force" ]; then
        echo -n "Proceed with deployment? [y/N]: "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo "Deployment cancelled."
            exit 0
        fi
    fi
    
    # Run deployment steps
    check_prerequisites
    install_docker
    install_docker_compose
    create_backup
    deploy_files
    configure_environment
    setup_monitoring
    configure_firewall
    deploy_services
    verify_deployment
    
    if [ "$SETUP_SSL" = "true" ]; then
        setup_ssl
    fi
    
    log "🎉 Deployment completed successfully!"
    log ""
    log "Next steps:"
    log "1. Configure the .env file on the server: ssh $DEPLOY_USER@$DEPLOY_HOST 'nano $DEPLOY_PATH/.env'"
    log "2. Add your SSH public key to authorized_keys for deployment access"
    log "3. Test the API: curl -H 'X-API-Key: your_api_key' http://$DEPLOY_HOST:8001/health"
    log "4. Access the application at: http://$DEPLOY_HOST:8001"
    
    if [ "$SETUP_SSL" = "true" ]; then
        log "5. Access via HTTPS: https://$DEPLOY_HOST"
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --ssl)
            SETUP_SSL="true"
            shift
            ;;
        --force)
            FORCE="true"
            shift
            ;;
        --host)
            DEPLOY_HOST="$2"
            shift 2
            ;;
        --user)
            DEPLOY_USER="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --ssl          Setup SSL certificate with Let's Encrypt"
            echo "  --force        Skip confirmation prompt"
            echo "  --host HOST    Target host (default: mcp.xplaincrypto.ai)"
            echo "  --user USER    SSH user (default: root)"
            echo "  -h, --help     Show this help message"
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main function
main "$@"
