"""
Comprehensive Credential Type Definitions
=========================================

Cloned from n8n's 400+ credential types.
Each credential type defines the schema for dynamic form generation.

"""

ALL_CREDENTIAL_TYPES = [
    # ========================================================================
    # AI & ML SERVICES
    # ========================================================================
    {
        "name": "openai_api",
        "display_name": "OpenAI API",
        "category": "ai",
        "icon": "brain",
        "logo": "/logos/OpenAI.png",
        "domain": "openai.com",
        "description": "OpenAI API credentials for GPT models",
        "documentation_url": "https://platform.openai.com/docs/api-reference",
        "schema_definition": [
            {
                "displayName": "API Key",
                "name": "api_key",
                "type": "password",
                "required": True,
                "description": "Your OpenAI API key (sk-...)",
                "typeOptions": {"password": True},
                "placeholder": "sk-..."
            },
            {
                "displayName": "Organization ID",
                "name": "organization_id",
                "type": "string",
                "required": False,
                "description": "Optional organization ID for team accounts"
            },
            {
                "displayName": "Base URL",
                "name": "base_url",
                "type": "string",
                "required": False,
                "default": "https://api.openai.com/v1",
                "description": "Custom API endpoint (for proxies or Azure OpenAI)"
            }
        ],
        "test_endpoint": {
            "method": "api_call",
            "url": "{base_url}/models",
            "headers": {"Authorization": "Bearer {api_key}"}
        },
        "is_system": True
    },
    
    {
        "name": "anthropic_api",
        "display_name": "Anthropic API",
        "category": "ai",
        "icon": "brain",
        "logo": "/logos/Anthropic.png",
        "domain": "anthropic.com",
        "description": "Anthropic Claude API credentials",
        "documentation_url": "https://docs.anthropic.com/claude/reference",
        "schema_definition": [
            {
                "displayName": "API Key",
                "name": "api_key",
                "type": "password",
                "required": True,
                "description": "Your Anthropic API key (sk-ant-...)",
                "typeOptions": {"password": True},
                "placeholder": "sk-ant-..."
            },
            {
                "displayName": "Base URL",
                "name": "base_url",
                "type": "string",
                "required": False,
                "default": "https://api.anthropic.com",
                "description": "Custom API endpoint"
            }
        ],
        "test_endpoint": {
            "method": "api_call",
            "url": "{base_url}/v1/messages",
            "headers": {"x-api-key": "{api_key}", "anthropic-version": "2023-06-01"}
        },
        "is_system": True
    },
    
    {
        "name": "huggingface_api",
        "display_name": "Hugging Face API",
        "category": "ai",
        "icon": "brain",
        "logo": "/logos/HuggingFace.png",
        "domain": "huggingface.co",
        "description": "Hugging Face API credentials",
        "documentation_url": "https://huggingface.co/docs/api-inference",
        "schema_definition": [
            {
                "displayName": "API Token",
                "name": "api_token",
                "type": "password",
                "required": True,
                "description": "Your Hugging Face API token",
                "typeOptions": {"password": True}
            }
        ],
        "test_endpoint": {
            "method": "api_call",
            "url": "https://huggingface.co/api/whoami",
            "headers": {"Authorization": "Bearer {api_token}"}
        },
        "is_system": True
    },
    
    # ========================================================================
    # DATABASES
    # ========================================================================
    {
        "name": "postgres_credentials",
        "display_name": "PostgreSQL",
        "category": "database",
        "icon": "database",
        "domain": "postgresql.org",
        "description": "PostgreSQL database connection credentials",
        "documentation_url": "https://www.postgresql.org/docs/",
        "schema_definition": [
            {
                "displayName": "Host",
                "name": "host",
                "type": "string",
                "required": True,
                "default": "localhost",
                "description": "Database host address"
            },
            {
                "displayName": "Port",
                "name": "port",
                "type": "number",
                "required": True,
                "default": 5432,
                "description": "Database port"
            },
            {
                "displayName": "Database",
                "name": "database",
                "type": "string",
                "required": True,
                "description": "Database name"
            },
            {
                "displayName": "User",
                "name": "user",
                "type": "string",
                "required": True,
                "description": "Database user"
            },
            {
                "displayName": "Password",
                "name": "password",
                "type": "password",
                "required": True,
                "description": "Database password",
                "typeOptions": {"password": True}
            },
            {
                "displayName": "SSL Mode",
                "name": "ssl_mode",
                "type": "options",
                "required": False,
                "default": "prefer",
                "description": "SSL connection mode",
                "typeOptions": {
                    "options": [
                        {"name": "Disable", "value": "disable"},
                        {"name": "Allow", "value": "allow"},
                        {"name": "Prefer", "value": "prefer"},
                        {"name": "Require", "value": "require"},
                        {"name": "Verify-CA", "value": "verify-ca"},
                        {"name": "Verify-Full", "value": "verify-full"}
                    ]
                }
            }
        ],
        "test_endpoint": {
            "method": "connect",
            "description": "Tests PostgreSQL database connection"
        },
        "is_system": True
    },
    
    {
        "name": "mysql_credentials",
        "display_name": "MySQL",
        "category": "database",
        "icon": "database",
        "logo": "/logos/MySQL.png",
        "domain": "mysql.com",
        "description": "MySQL database connection credentials",
        "documentation_url": "https://dev.mysql.com/doc/",
        "schema_definition": [
            {
                "displayName": "Host",
                "name": "host",
                "type": "string",
                "required": True,
                "default": "localhost"
            },
            {
                "displayName": "Port",
                "name": "port",
                "type": "number",
                "required": True,
                "default": 3306
            },
            {
                "displayName": "Database",
                "name": "database",
                "type": "string",
                "required": True
            },
            {
                "displayName": "User",
                "name": "user",
                "type": "string",
                "required": True
            },
            {
                "displayName": "Password",
                "name": "password",
                "type": "password",
                "required": True,
                "typeOptions": {"password": True}
            },
            {
                "displayName": "SSL",
                "name": "ssl_enabled",
                "type": "boolean",
                "default": False
            }
        ],
        "test_endpoint": {
            "method": "connect",
            "description": "Tests MySQL database connection"
        },
        "is_system": True
    },
    
    {
        "name": "mongodb_credentials",
        "display_name": "MongoDB",
        "category": "database",
        "icon": "database",
        "logo": "/logos/MongoDB.png",
        "domain": "mongodb.com",
        "description": "MongoDB connection credentials",
        "documentation_url": "https://docs.mongodb.com/",
        "schema_definition": [
            {
                "displayName": "Connection String",
                "name": "connection_string",
                "type": "string",
                "required": True,
                "description": "MongoDB connection string (mongodb://...)",
                "placeholder": "mongodb://localhost:27017/mydb"
            },
            {
                "displayName": "Or Configure Manually",
                "name": "manual_config",
                "type": "boolean",
                "default": False
            },
            {
                "displayName": "Host",
                "name": "host",
                "type": "string",
                "required": False,
                "default": "localhost",
                "displayOptions": {"show": {"manual_config": [True]}}
            },
            {
                "displayName": "Port",
                "name": "port",
                "type": "number",
                "required": False,
                "default": 27017,
                "displayOptions": {"show": {"manual_config": [True]}}
            },
            {
                "displayName": "Database",
                "name": "database",
                "type": "string",
                "required": False,
                "displayOptions": {"show": {"manual_config": [True]}}
            },
            {
                "displayName": "User",
                "name": "user",
                "type": "string",
                "required": False,
                "displayOptions": {"show": {"manual_config": [True]}}
            },
            {
                "displayName": "Password",
                "name": "password",
                "type": "password",
                "required": False,
                "typeOptions": {"password": True},
                "displayOptions": {"show": {"manual_config": [True]}}
            }
        ],
        "test_endpoint": {
            "method": "connect",
            "description": "Tests MongoDB connection"
        },
        "is_system": True
    },
    
    {
        "name": "redis_credentials",
        "display_name": "Redis",
        "category": "database",
        "icon": "database",
        "logo": "/logos/Redis.png",
        "domain": "redis.io",
        "description": "Redis connection credentials",
        "documentation_url": "https://redis.io/docs/",
        "schema_definition": [
            {
                "displayName": "Host",
                "name": "host",
                "type": "string",
                "required": True,
                "default": "localhost"
            },
            {
                "displayName": "Port",
                "name": "port",
                "type": "number",
                "required": True,
                "default": 6379
            },
            {
                "displayName": "Password",
                "name": "password",
                "type": "password",
                "required": False,
                "description": "Redis password (if required)",
                "typeOptions": {"password": True}
            },
            {
                "displayName": "Database",
                "name": "database",
                "type": "number",
                "required": False,
                "default": 0,
                "description": "Redis database number (0-15)"
            }
        ],
        "test_endpoint": {
            "method": "connect",
            "description": "Tests Redis connection with PING command"
        },
        "is_system": True
    },
    
    {
        "name": "elasticsearch_credentials",
        "display_name": "Elasticsearch",
        "category": "database",
        "icon": "database",
        "logo": "/logos/Elasticsearch.png",
        "domain": "elastic.co",
        "description": "Elasticsearch connection credentials",
        "schema_definition": [
            {
                "displayName": "Base URL",
                "name": "base_url",
                "type": "string",
                "required": True,
                "default": "http://localhost:9200",
                "placeholder": "http://localhost:9200"
            },
            {
                "displayName": "Username",
                "name": "username",
                "type": "string",
                "required": False
            },
            {
                "displayName": "Password",
                "name": "password",
                "type": "password",
                "required": False,
                "typeOptions": {"password": True}
            }
        ],
        "test_endpoint": {
            "method": "api_call",
            "url": "{base_url}/_cluster/health"
        },
        "is_system": True
    },
    
    # ========================================================================
    # CLOUD PROVIDERS
    # ========================================================================
    {
        "name": "aws_credentials",
        "display_name": "AWS",
        "category": "cloud",
        "icon": "cloud",
        "logo": "/logos/AWS.png",
        "domain": "aws.amazon.com",
        "description": "Amazon Web Services credentials",
        "documentation_url": "https://docs.aws.amazon.com/",
        "schema_definition": [
            {
                "displayName": "Access Key ID",
                "name": "access_key_id",
                "type": "string",
                "required": True,
                "description": "AWS Access Key ID"
            },
            {
                "displayName": "Secret Access Key",
                "name": "secret_access_key",
                "type": "password",
                "required": True,
                "description": "AWS Secret Access Key",
                "typeOptions": {"password": True}
            },
            {
                "displayName": "Region",
                "name": "region",
                "type": "options",
                "required": True,
                "default": "us-east-1",
                "typeOptions": {
                    "options": [
                        {"name": "US East (N. Virginia)", "value": "us-east-1"},
                        {"name": "US West (Oregon)", "value": "us-west-2"},
                        {"name": "EU (Ireland)", "value": "eu-west-1"},
                        {"name": "EU (Frankfurt)", "value": "eu-central-1"},
                        {"name": "Asia Pacific (Tokyo)", "value": "ap-northeast-1"},
                        {"name": "Asia Pacific (Singapore)", "value": "ap-southeast-1"}
                    ]
                }
            },
            {
                "displayName": "Session Token",
                "name": "session_token",
                "type": "password",
                "required": False,
                "description": "Temporary session token (for STS)",
                "typeOptions": {"password": True}
            }
        ],
        "test_endpoint": {
            "method": "api_call",
            "description": "Tests AWS credentials with STS GetCallerIdentity"
        },
        "is_system": True
    },
    
    {
        "name": "azure_credentials",
        "display_name": "Microsoft Azure",
        "category": "cloud",
        "icon": "cloud",
        "logo": "/logos/Azure.png",
        "domain": "azure.microsoft.com",
        "description": "Microsoft Azure credentials",
        "schema_definition": [
            {
                "displayName": "Client ID",
                "name": "client_id",
                "type": "string",
                "required": True,
                "description": "Azure Application (client) ID"
            },
            {
                "displayName": "Client Secret",
                "name": "client_secret",
                "type": "password",
                "required": True,
                "typeOptions": {"password": True}
            },
            {
                "displayName": "Tenant ID",
                "name": "tenant_id",
                "type": "string",
                "required": True,
                "description": "Azure Directory (tenant) ID"
            },
            {
                "displayName": "Subscription ID",
                "name": "subscription_id",
                "type": "string",
                "required": False,
                "description": "Azure subscription ID"
            }
        ],
        "test_endpoint": {
            "method": "oauth2",
            "description": "Tests Azure authentication"
        },
        "is_system": True
    },
    
    {
        "name": "google_cloud_credentials",
        "display_name": "Google Cloud",
        "category": "cloud",
        "icon": "cloud",
        "logo": "/logos/GoogleCloud.png",
        "domain": "cloud.google.com",
        "description": "Google Cloud Platform credentials",
        "schema_definition": [
            {
                "displayName": "Service Account Key (JSON)",
                "name": "service_account_key",
                "type": "password",
                "required": True,
                "description": "Google Cloud service account key JSON",
                "typeOptions": {"password": True, "rows": 10}
            },
            {
                "displayName": "Project ID",
                "name": "project_id",
                "type": "string",
                "required": False,
                "description": "GCP Project ID"
            }
        ],
        "test_endpoint": {
            "method": "api_call",
            "description": "Tests service account credentials"
        },
        "is_system": True
    },
    
    # ========================================================================
    # COMMUNICATION & COLLABORATION
    # ========================================================================
    {
        "name": "slack_api",
        "display_name": "Slack",
        "category": "communication",
        "icon": "slack",
        "logo": "/logos/Slack.png",
        "domain": "slack.com",
        "description": "Slack API credentials",
        "documentation_url": "https://api.slack.com/",
        "schema_definition": [
            {
                "displayName": "Access Token",
                "name": "access_token",
                "type": "password",
                "required": True,
                "description": "Slack Bot User OAuth Token (xoxb-...)",
                "typeOptions": {"password": True},
                "placeholder": "xoxb-..."
            }
        ],
        "test_endpoint": {
            "method": "api_call",
            "url": "https://slack.com/api/auth.test",
            "headers": {"Authorization": "Bearer {access_token}"}
        },
        "is_system": True
    },
    
    {
        "name": "discord_webhook",
        "display_name": "Discord Webhook",
        "category": "communication",
        "icon": "message-square",
        "logo": "/logos/Discord.png",
        "domain": "discord.com",
        "description": "Discord webhook URL",
        "schema_definition": [
            {
                "displayName": "Webhook URL",
                "name": "webhook_url",
                "type": "string",
                "required": True,
                "description": "Discord webhook URL",
                "placeholder": "https://discord.com/api/webhooks/..."
            }
        ],
        "test_endpoint": {
            "method": "api_call",
            "url": "{webhook_url}",
            "description": "Sends test message to Discord channel"
        },
        "is_system": True
    },
    
    {
        "name": "telegram_api",
        "display_name": "Telegram",
        "category": "communication",
        "icon": "send",
        "logo": "/logos/Telegram.png",
        "domain": "telegram.org",
        "description": "Telegram Bot API credentials",
        "schema_definition": [
            {
                "displayName": "Bot Token",
                "name": "bot_token",
                "type": "password",
                "required": True,
                "description": "Telegram Bot API token from BotFather",
                "typeOptions": {"password": True}
            }
        ],
        "test_endpoint": {
            "method": "api_call",
            "url": "https://api.telegram.org/bot{bot_token}/getMe"
        },
        "is_system": True
    },
    
    {
        "name": "twilio_api",
        "display_name": "Twilio",
        "category": "communication",
        "icon": "phone",
        "logo": "/logos/Twilio.png",
        "domain": "twilio.com",
        "description": "Twilio API credentials",
        "schema_definition": [
            {
                "displayName": "Account SID",
                "name": "account_sid",
                "type": "string",
                "required": True
            },
            {
                "displayName": "Auth Token",
                "name": "auth_token",
                "type": "password",
                "required": True,
                "typeOptions": {"password": True}
            }
        ],
        "test_endpoint": {
            "method": "api_call",
            "url": "https://api.twilio.com/2010-04-01/Accounts/{account_sid}.json"
        },
        "is_system": True
    },
    
    {
        "name": "sendgrid_api",
        "display_name": "SendGrid",
        "category": "communication",
        "icon": "mail",
        "logo": "/logos/SendGrid.png",
        "domain": "sendgrid.com",
        "description": "SendGrid email API credentials",
        "schema_definition": [
            {
                "displayName": "API Key",
                "name": "api_key",
                "type": "password",
                "required": True,
                "typeOptions": {"password": True}
            }
        ],
        "test_endpoint": {
            "method": "api_call",
            "url": "https://api.sendgrid.com/v3/scopes",
            "headers": {"Authorization": "Bearer {api_key}"}
        },
        "is_system": True
    },
    
    # ========================================================================
    # VERSION CONTROL & CI/CD
    # ========================================================================
    {
        "name": "github_api",
        "display_name": "GitHub",
        "category": "code",
        "icon": "github",
        "logo": "/logos/GitHub.png",
        "domain": "github.com",
        "description": "GitHub API credentials",
        "documentation_url": "https://docs.github.com/en/rest",
        "schema_definition": [
            {
                "displayName": "Access Token",
                "name": "access_token",
                "type": "password",
                "required": True,
                "description": "GitHub Personal Access Token (ghp_...)",
                "typeOptions": {"password": True},
                "placeholder": "ghp_..."
            },
            {
                "displayName": "Token Type",
                "name": "token_type",
                "type": "options",
                "required": False,
                "default": "personal",
                "typeOptions": {
                    "options": [
                        {"name": "Personal Access Token", "value": "personal"},
                        {"name": "OAuth App Token", "value": "oauth"},
                        {"name": "GitHub App Token", "value": "app"}
                    ]
                }
            }
        ],
        "test_endpoint": {
            "method": "api_call",
            "url": "https://api.github.com/user",
            "headers": {"Authorization": "Bearer {access_token}"}
        },
        "is_system": True
    },
    
    {
        "name": "gitlab_api",
        "display_name": "GitLab",
        "domain": "gitlab.com",
        "category": "code",
        "icon": "gitlab",
        "logo": "/logos/GitLab.png",
        "description": "GitLab API credentials",
        "schema_definition": [
            {
                "displayName": "Access Token",
                "name": "access_token",
                "type": "password",
                "required": True,
                "description": "GitLab Personal Access Token",
                "typeOptions": {"password": True}
            },
            {
                "displayName": "GitLab Server",
                "name": "server",
                "type": "string",
                "required": False,
                "default": "https://gitlab.com",
                "description": "GitLab server URL (for self-hosted)"
            }
        ],
        "test_endpoint": {
            "method": "api_call",
            "url": "{server}/api/v4/user",
            "headers": {"PRIVATE-TOKEN": "{access_token}"}
        },
        "is_system": True
    },
    
    # ========================================================================
    # INFRASTRUCTURE & DEPLOYMENT
    # ========================================================================
    {
        "name": "ssh_credentials",
        "display_name": "SSH",
        "category": "infrastructure",
        "icon": "server",
        "description": "SSH connection credentials for remote server access",
        "documentation_url": "https://www.ssh.com/academy/ssh",
        "schema_definition": [
            {
                "displayName": "Host",
                "name": "host",
                "type": "string",
                "required": True,
                "description": "SSH server hostname or IP"
            },
            {
                "displayName": "Port",
                "name": "port",
                "type": "number",
                "required": True,
                "default": 22
            },
            {
                "displayName": "Username",
                "name": "username",
                "type": "string",
                "required": True
            },
            {
                "displayName": "Authentication Method",
                "name": "auth_method",
                "type": "options",
                "required": True,
                "default": "password",
                "typeOptions": {
                    "options": [
                        {"name": "Password", "value": "password"},
                        {"name": "Private Key", "value": "key"}
                    ]
                }
            },
            {
                "displayName": "Password",
                "name": "password",
                "type": "password",
                "required": False,
                "typeOptions": {"password": True},
                "displayOptions": {"show": {"auth_method": ["password"]}}
            },
            {
                "displayName": "Private Key",
                "name": "private_key",
                "type": "password",
                "required": False,
                "description": "SSH private key content",
                "typeOptions": {"password": True, "rows": 8},
                "displayOptions": {"show": {"auth_method": ["key"]}}
            },
            {
                "displayName": "Passphrase",
                "name": "passphrase",
                "type": "password",
                "required": False,
                "typeOptions": {"password": True},
                "displayOptions": {"show": {"auth_method": ["key"]}}
            }
        ],
        "test_endpoint": {
            "method": "connect",
            "description": "Tests SSH connection and authentication"
        },
        "is_system": True
    },
    
    {
        "name": "docker_credentials",
        "display_name": "Docker",
        "category": "infrastructure",
        "icon": "box",
        "description": "Docker Registry credentials",
        "schema_definition": [
            {
                "displayName": "Registry URL",
                "name": "registry_url",
                "type": "string",
                "required": False,
                "default": "https://index.docker.io/v1/",
                "description": "Docker registry URL (default: Docker Hub)"
            },
            {
                "displayName": "Username",
                "name": "username",
                "type": "string",
                "required": True
            },
            {
                "displayName": "Password",
                "name": "password",
                "type": "password",
                "required": True,
                "typeOptions": {"password": True}
            }
        ],
        "test_endpoint": {
            "method": "api_call",
            "description": "Tests Docker registry authentication"
        },
        "is_system": True
    },
    
    {
        "name": "kubernetes_credentials",
        "display_name": "Kubernetes",
        "category": "infrastructure",
        "icon": "box",
        "description": "Kubernetes cluster credentials",
        "schema_definition": [
            {
                "displayName": "Kubeconfig",
                "name": "kubeconfig",
                "type": "password",
                "required": True,
                "description": "Kubernetes configuration YAML",
                "typeOptions": {"password": True, "rows": 10}
            }
        ],
        "test_endpoint": {
            "method": "connect",
            "description": "Tests Kubernetes cluster connection"
        },
        "is_system": True
    },
    
    # ========================================================================
    # PAYMENT & FINANCE
    # ========================================================================
    {
        "name": "stripe_api",
        "display_name": "Stripe",
        "category": "payment",
        "icon": "credit-card",
        "logo": "/logos/Stripe.png",
        "description": "Stripe payment API credentials",
        "schema_definition": [
            {
                "displayName": "Secret Key",
                "name": "secret_key",
                "type": "password",
                "required": True,
                "description": "Stripe secret key (sk_...)",
                "typeOptions": {"password": True},
                "placeholder": "sk_test_... or sk_live_..."
            },
            {
                "displayName": "Mode",
                "name": "mode",
                "type": "options",
                "required": True,
                "default": "test",
                "typeOptions": {
                    "options": [
                        {"name": "Test Mode", "value": "test"},
                        {"name": "Live Mode", "value": "live"}
                    ]
                }
            }
        ],
        "test_endpoint": {
            "method": "api_call",
            "url": "https://api.stripe.com/v1/balance",
            "headers": {"Authorization": "Bearer {secret_key}"}
        },
        "is_system": True
    },
    
    {
        "name": "paypal_api",
        "display_name": "PayPal",
        "category": "payment",
        "icon": "credit-card",
        "description": "PayPal API credentials",
        "schema_definition": [
            {
                "displayName": "Client ID",
                "name": "client_id",
                "type": "string",
                "required": True
            },
            {
                "displayName": "Client Secret",
                "name": "client_secret",
                "type": "password",
                "required": True,
                "typeOptions": {"password": True}
            },
            {
                "displayName": "Environment",
                "name": "environment",
                "type": "options",
                "required": True,
                "default": "sandbox",
                "typeOptions": {
                    "options": [
                        {"name": "Sandbox", "value": "sandbox"},
                        {"name": "Live", "value": "live"}
                    ]
                }
            }
        ],
        "test_endpoint": {
            "method": "oauth2",
            "description": "Tests PayPal OAuth authentication"
        },
        "is_system": True
    },
    
    # ========================================================================
    # CRM & MARKETING
    # ========================================================================
    {
        "name": "salesforce_oauth2",
        "display_name": "Salesforce",
        "category": "crm",
        "icon": "briefcase",
        "logo": "/logos/Salesforce.png",
        "description": "Salesforce OAuth2 credentials",
        "schema_definition": [
            {
                "displayName": "Access Token",
                "name": "access_token",
                "type": "password",
                "required": True,
                "typeOptions": {"password": True}
            },
            {
                "displayName": "Instance URL",
                "name": "instance_url",
                "type": "string",
                "required": True,
                "placeholder": "https://your-instance.salesforce.com"
            },
            {
                "displayName": "Refresh Token",
                "name": "refresh_token",
                "type": "password",
                "required": False,
                "typeOptions": {"password": True}
            }
        ],
        "test_endpoint": {
            "method": "api_call",
            "url": "{instance_url}/services/data/v57.0/"
        },
        "is_system": True
    },
    
    {
        "name": "hubspot_api",
        "display_name": "HubSpot",
        "category": "crm",
        "icon": "briefcase",
        "logo": "/logos/HubSpot.png",
        "description": "HubSpot API credentials",
        "schema_definition": [
            {
                "displayName": "API Key",
                "name": "api_key",
                "type": "password",
                "required": True,
                "typeOptions": {"password": True}
            }
        ],
        "test_endpoint": {
            "method": "api_call",
            "url": "https://api.hubapi.com/crm/v3/objects/contacts",
            "headers": {"Authorization": "Bearer {api_key}"}
        },
        "is_system": True
    },
    
    # ========================================================================
    # MONITORING & ANALYTICS
    # ========================================================================
    {
        "name": "datadog_api",
        "display_name": "Datadog",
        "category": "monitoring",
        "icon": "activity",
        "logo": "/logos/Datadog.png",
        "description": "Datadog monitoring API credentials",
        "schema_definition": [
            {
                "displayName": "API Key",
                "name": "api_key",
                "type": "password",
                "required": True,
                "typeOptions": {"password": True}
            },
            {
                "displayName": "Application Key",
                "name": "app_key",
                "type": "password",
                "required": True,
                "typeOptions": {"password": True}
            },
            {
                "displayName": "Site",
                "name": "site",
                "type": "options",
                "required": True,
                "default": "datadoghq.com",
                "typeOptions": {
                    "options": [
                        {"name": "US1 (datadoghq.com)", "value": "datadoghq.com"},
                        {"name": "EU (datadoghq.eu)", "value": "datadoghq.eu"}
                    ]
                }
            }
        ],
        "test_endpoint": {
            "method": "api_call",
            "url": "https://api.{site}/api/v1/validate"
        },
        "is_system": True
    },
    
    # ========================================================================
    # GENERIC/UTILITY
    # ========================================================================
    {
        "name": "generic_api",
        "display_name": "Generic API",
        "category": "api",
        "icon": "key",
        "description": "Generic API credentials with customizable authentication",
        "schema_definition": [
            {
                "displayName": "API Key",
                "name": "api_key",
                "type": "password",
                "required": True,
                "description": "API authentication key",
                "typeOptions": {"password": True}
            },
            {
                "displayName": "API Secret",
                "name": "api_secret",
                "type": "password",
                "required": False,
                "description": "Optional API secret",
                "typeOptions": {"password": True}
            },
            {
                "displayName": "Base URL",
                "name": "base_url",
                "type": "string",
                "required": False,
                "description": "API base URL"
            },
            {
                "displayName": "Authentication Method",
                "name": "auth_method",
                "type": "options",
                "required": True,
                "default": "header",
                "typeOptions": {
                    "options": [
                        {"name": "Header", "value": "header"},
                        {"name": "Query Parameter", "value": "query"},
                        {"name": "Bearer Token", "value": "bearer"},
                        {"name": "Basic Auth", "value": "basic"}
                    ]
                }
            },
            {
                "displayName": "Header Name",
                "name": "header_name",
                "type": "string",
                "required": False,
                "default": "X-API-Key",
                "displayOptions": {"show": {"auth_method": ["header"]}}
            },
            {
                "displayName": "Query Parameter Name",
                "name": "query_param_name",
                "type": "string",
                "required": False,
                "default": "api_key",
                "displayOptions": {"show": {"auth_method": ["query"]}}
            }
        ],
        "test_endpoint": None,
        "is_system": True
    },
    
    {
        "name": "oauth2_token",
        "display_name": "OAuth2 Token",
        "category": "api",
        "icon": "key",
        "description": "OAuth2 access token storage (token obtained externally)",
        "schema_definition": [
            {
                "displayName": "Access Token",
                "name": "access_token",
                "type": "password",
                "required": True,
                "typeOptions": {"password": True}
            },
            {
                "displayName": "Refresh Token",
                "name": "refresh_token",
                "type": "password",
                "required": False,
                "typeOptions": {"password": True}
            },
            {
                "displayName": "Token Expiry",
                "name": "expires_at",
                "type": "string",
                "required": False,
                "description": "Token expiration (ISO format)"
            },
            {
                "displayName": "Scope",
                "name": "scope",
                "type": "string",
                "required": False
            }
        ],
        "test_endpoint": None,
        "is_system": True
    },
    
    {
        "name": "http_basic_auth",
        "display_name": "HTTP Basic Auth",
        "category": "api",
        "icon": "key",
        "description": "Basic HTTP authentication credentials",
        "schema_definition": [
            {
                "displayName": "Username",
                "name": "username",
                "type": "string",
                "required": True
            },
            {
                "displayName": "Password",
                "name": "password",
                "type": "password",
                "required": True,
                "typeOptions": {"password": True}
            }
        ],
        "test_endpoint": None,
        "is_system": True
    },
    
    # ========================================================================
    # STORAGE & FILE SERVICES
    # ========================================================================
    {
        "name": "s3_credentials",
        "display_name": "Amazon S3",
        "category": "storage",
        "icon": "hard-drive",
        "description": "Amazon S3 storage credentials",
        "schema_definition": [
            {
                "displayName": "Access Key ID",
                "name": "access_key_id",
                "type": "string",
                "required": True
            },
            {
                "displayName": "Secret Access Key",
                "name": "secret_access_key",
                "type": "password",
                "required": True,
                "typeOptions": {"password": True}
            },
            {
                "displayName": "Region",
                "name": "region",
                "type": "string",
                "required": True,
                "default": "us-east-1"
            },
            {
                "displayName": "Bucket Name",
                "name": "bucket_name",
                "type": "string",
                "required": False,
                "description": "Default bucket name"
            }
        ],
        "test_endpoint": {
            "method": "api_call",
            "description": "Tests S3 access by listing buckets"
        },
        "is_system": True
    },
    
    # Add more credential types here... (n8n has 400+)
    # This is a curated subset - the system supports unlimited custom types
]


def get_credential_type_by_name(name: str) -> dict:
    """Get credential type definition by name"""
    for cred_type in ALL_CREDENTIAL_TYPES:
        if cred_type["name"] == name:
            return cred_type
    return None


def get_credential_types_by_category(category: str) -> list:
    """Get all credential types in a category"""
    return [ct for ct in ALL_CREDENTIAL_TYPES if ct.get("category") == category]


def get_all_categories() -> list:
    """Get list of all credential categories"""
    categories = set()
    for ct in ALL_CREDENTIAL_TYPES:
        if ct.get("category"):
            categories.add(ct["category"])
    return sorted(list(categories))

