"""
n8n-Style Credential Testing Implementation
==========================================

Based on n8n's approach to credential testing:
- Make actual API calls to verify credentials work
- Test different authentication methods
- Provide detailed error messages
- Support various credential types
"""

import asyncio
import aiohttp
import asyncpg
import redis
import paramiko
import json
import logging
from typing import Dict, Any, Optional, Union
from datetime import datetime

logger = logging.getLogger(__name__)

class CredentialTester:
    """Test credentials using n8n-style validation"""
    
    def __init__(self):
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_credential(self, credential_type: str, credential_data: Dict[str, Any]) -> Dict[str, Any]:
        """Test a credential based on its type"""
        
        test_methods = {
            'openai_api': self._test_openai,
            'anthropic_api': self._test_anthropic,
            'postgres_credentials': self._test_postgres,
            'redis_credentials': self._test_redis,
            'github_api': self._test_github,
            'ssh_credentials': self._test_ssh,
            'generic_api': self._test_generic_api,
            'slack_api': self._test_slack,
            'discord_webhook': self._test_discord_webhook,
            'telegram_api': self._test_telegram,
            'aws_credentials': self._test_aws,
            'azure_credentials': self._test_azure,
            'google_cloud_credentials': self._test_google_cloud,
            'stripe_api': self._test_stripe,
            'paypal_api': self._test_paypal,
            'salesforce_oauth2': self._test_salesforce,
            'hubspot_api': self._test_hubspot,
            'datadog_api': self._test_datadog,
            'elasticsearch_credentials': self._test_elasticsearch,
            'mongodb_credentials': self._test_mongodb,
            'mysql_credentials': self._test_mysql,
            'docker_credentials': self._test_docker,
            'kubernetes_credentials': self._test_kubernetes,
            'sendgrid_api': self._test_sendgrid,
            'twilio_api': self._test_twilio,
            'gitlab_api': self._test_gitlab,
            'huggingface_api': self._test_huggingface,
            'oauth2_token': self._test_oauth2_token,
            'http_basic_auth': self._test_http_basic_auth,
            's3_credentials': self._test_s3
        }
        
        if credential_type not in test_methods:
            return {
                'success': False,
                'message': f'No test method available for credential type: {credential_type}',
                'details': {'error': 'unsupported_type'}
            }
        
        try:
            return await test_methods[credential_type](credential_data)
        except Exception as e:
            logger.error(f"Credential test failed for {credential_type}: {e}")
            return {
                'success': False,
                'message': f'Test failed: {str(e)}',
                'details': {'error': str(e), 'type': type(e).__name__}
            }
    
    async def _test_openai(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Test OpenAI API credentials"""
        api_key = data.get('api_key')
        base_url = data.get('base_url', 'https://api.openai.com/v1')
        
        if not api_key:
            return {'success': False, 'message': 'API key is required'}
        
        url = f"{base_url}/models"
        headers = {'Authorization': f'Bearer {api_key}'}
        
        async with self.session.get(url, headers=headers) as response:
            if response.status == 200:
                models = await response.json()
                return {
                    'success': True,
                    'message': f'OpenAI API connection successful. Found {len(models.get("data", []))} models.',
                    'details': {
                        'models_count': len(models.get("data", [])),
                        'organization': models.get('organization', 'Unknown')
                    }
                }
            else:
                error_text = await response.text()
                return {
                    'success': False,
                    'message': f'OpenAI API test failed: {response.status} - {error_text}',
                    'details': {'status_code': response.status, 'error': error_text}
                }
    
    async def _test_anthropic(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Test Anthropic API credentials"""
        api_key = data.get('api_key')
        base_url = data.get('base_url', 'https://api.anthropic.com')
        
        if not api_key:
            return {'success': False, 'message': 'API key is required'}
        
        url = f"{base_url}/v1/messages"
        headers = {
            'x-api-key': api_key,
            'anthropic-version': '2023-06-01',
            'content-type': 'application/json'
        }
        
        # Test with a simple message
        payload = {
            'model': 'claude-3-haiku-20240307',
            'max_tokens': 10,
            'messages': [{'role': 'user', 'content': 'Hello'}]
        }
        
        async with self.session.post(url, headers=headers, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                return {
                    'success': True,
                    'message': 'Anthropic API connection successful',
                    'details': {
                        'model': result.get('model', 'Unknown'),
                        'usage': result.get('usage', {})
                    }
                }
            else:
                error_text = await response.text()
                return {
                    'success': False,
                    'message': f'Anthropic API test failed: {response.status} - {error_text}',
                    'details': {'status_code': response.status, 'error': error_text}
                }
    
    async def _test_postgres(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Test PostgreSQL connection"""
        host = data.get('host', 'localhost')
        port = data.get('port', 5432)
        database = data.get('database')
        user = data.get('user')
        password = data.get('password')
        
        if not all([host, database, user, password]):
            return {'success': False, 'message': 'Host, database, user, and password are required'}
        
        try:
            conn = await asyncpg.connect(
                host=host,
                port=int(port),
                database=database,
                user=user,
                password=password
            )
            
            # Test query
            version = await conn.fetchval('SELECT version()')
            await conn.close()
            
            return {
                'success': True,
                'message': f'PostgreSQL connection successful',
                'details': {
                    'version': version,
                    'host': host,
                    'database': database
                }
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'PostgreSQL connection failed: {str(e)}',
                'details': {'error': str(e)}
            }
    
    async def _test_redis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Test Redis connection"""
        host = data.get('host', 'localhost')
        port = data.get('port', 6379)
        password = data.get('password')
        database = data.get('database', 0)
        
        try:
            r = redis.Redis(
                host=host,
                port=int(port),
                password=password,
                db=int(database),
                decode_responses=True
            )
            
            # Test with PING
            pong = r.ping()
            info = r.info()
            
            return {
                'success': True,
                'message': 'Redis connection successful',
                'details': {
                    'ping': pong,
                    'version': info.get('redis_version', 'Unknown'),
                    'memory_used': info.get('used_memory_human', 'Unknown')
                }
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Redis connection failed: {str(e)}',
                'details': {'error': str(e)}
            }
    
    async def _test_github(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Test GitHub API credentials"""
        access_token = data.get('access_token')
        
        if not access_token:
            return {'success': False, 'message': 'Access token is required'}
        
        url = 'https://api.github.com/user'
        headers = {'Authorization': f'Bearer {access_token}'}
        
        async with self.session.get(url, headers=headers) as response:
            if response.status == 200:
                user_data = await response.json()
                return {
                    'success': True,
                    'message': f'GitHub API connection successful. Authenticated as {user_data.get("login", "Unknown")}',
                    'details': {
                        'username': user_data.get('login'),
                        'name': user_data.get('name'),
                        'email': user_data.get('email'),
                        'public_repos': user_data.get('public_repos', 0)
                    }
                }
            else:
                error_text = await response.text()
                return {
                    'success': False,
                    'message': f'GitHub API test failed: {response.status} - {error_text}',
                    'details': {'status_code': response.status, 'error': error_text}
                }
    
    async def _test_ssh(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Test SSH connection"""
        host = data.get('host')
        port = data.get('port', 22)
        username = data.get('username')
        password = data.get('password')
        private_key = data.get('private_key')
        auth_method = data.get('auth_method', 'password')
        
        if not all([host, username]):
            return {'success': False, 'message': 'Host and username are required'}
        
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            if auth_method == 'password' and password:
                ssh.connect(host, port=int(port), username=username, password=password)
            elif auth_method == 'key' and private_key:
                key = paramiko.RSAKey.from_private_key_file(private_key) if private_key.startswith('/') else paramiko.RSAKey.from_private_key(io.StringIO(private_key))
                ssh.connect(host, port=int(port), username=username, pkey=key)
            else:
                return {'success': False, 'message': 'Invalid authentication method or missing credentials'}
            
            # Test command
            stdin, stdout, stderr = ssh.exec_command('echo "SSH test successful"')
            output = stdout.read().decode().strip()
            ssh.close()
            
            return {
                'success': True,
                'message': 'SSH connection successful',
                'details': {
                    'host': host,
                    'port': port,
                    'username': username,
                    'test_output': output
                }
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'SSH connection failed: {str(e)}',
                'details': {'error': str(e)}
            }
    
    async def _test_generic_api(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Test generic API credentials"""
        api_key = data.get('api_key')
        base_url = data.get('base_url')
        auth_method = data.get('auth_method', 'header')
        header_name = data.get('header_name', 'X-API-Key')
        
        if not api_key:
            return {'success': False, 'message': 'API key is required'}
        
        if not base_url:
            return {'success': False, 'message': 'Base URL is required for testing'}
        
        headers = {}
        params = {}
        
        if auth_method == 'header':
            headers[header_name] = api_key
        elif auth_method == 'query':
            params['api_key'] = api_key
        elif auth_method == 'bearer':
            headers['Authorization'] = f'Bearer {api_key}'
        
        try:
            async with self.session.get(base_url, headers=headers, params=params) as response:
                if response.status in [200, 201, 202]:
                    return {
                        'success': True,
                        'message': f'Generic API connection successful (Status: {response.status})',
                        'details': {
                            'status_code': response.status,
                            'auth_method': auth_method,
                            'base_url': base_url
                        }
                    }
                else:
                    error_text = await response.text()
                    return {
                        'success': False,
                        'message': f'Generic API test failed: {response.status} - {error_text}',
                        'details': {'status_code': response.status, 'error': error_text}
                    }
        except Exception as e:
            return {
                'success': False,
                'message': f'Generic API test failed: {str(e)}',
                'details': {'error': str(e)}
            }
    
    # Additional test methods for other credential types...
    async def _test_slack(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Test Slack API credentials"""
        access_token = data.get('access_token')
        
        if not access_token:
            return {'success': False, 'message': 'Access token is required'}
        
        url = 'https://slack.com/api/auth.test'
        headers = {'Authorization': f'Bearer {access_token}'}
        
        async with self.session.get(url, headers=headers) as response:
            if response.status == 200:
                result = await response.json()
                if result.get('ok'):
                    return {
                        'success': True,
                        'message': f'Slack API connection successful. Bot: {result.get("user", "Unknown")}',
                        'details': result
                    }
                else:
                    return {
                        'success': False,
                        'message': f'Slack API test failed: {result.get("error", "Unknown error")}',
                        'details': result
                    }
            else:
                error_text = await response.text()
                return {
                    'success': False,
                    'message': f'Slack API test failed: {response.status} - {error_text}',
                    'details': {'status_code': response.status, 'error': error_text}
                }
    
    async def _test_discord_webhook(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Test Discord webhook"""
        webhook_url = data.get('webhook_url')
        
        if not webhook_url:
            return {'success': False, 'message': 'Webhook URL is required'}
        
        payload = {
            'content': '🔧 Credential test from Automatos AI Platform',
            'embeds': [{
                'title': 'Credential Test',
                'description': 'This is a test message to verify webhook connectivity',
                'color': 0x00ff00,
                'timestamp': datetime.utcnow().isoformat()
            }]
        }
        
        async with self.session.post(webhook_url, json=payload) as response:
            if response.status == 204:
                return {
                    'success': True,
                    'message': 'Discord webhook test successful',
                    'details': {'status_code': response.status}
                }
            else:
                error_text = await response.text()
                return {
                    'success': False,
                    'message': f'Discord webhook test failed: {response.status} - {error_text}',
                    'details': {'status_code': response.status, 'error': error_text}
                }
    
    async def _test_telegram(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Test Telegram Bot API"""
        bot_token = data.get('bot_token')
        
        if not bot_token:
            return {'success': False, 'message': 'Bot token is required'}
        
        url = f'https://api.telegram.org/bot{bot_token}/getMe'
        
        async with self.session.get(url) as response:
            if response.status == 200:
                result = await response.json()
                if result.get('ok'):
                    bot_info = result.get('result', {})
                    return {
                        'success': True,
                        'message': f'Telegram Bot API connection successful. Bot: @{bot_info.get("username", "Unknown")}',
                        'details': bot_info
                    }
                else:
                    return {
                        'success': False,
                        'message': f'Telegram API test failed: {result.get("description", "Unknown error")}',
                        'details': result
                    }
            else:
                error_text = await response.text()
                return {
                    'success': False,
                    'message': f'Telegram API test failed: {response.status} - {error_text}',
                    'details': {'status_code': response.status, 'error': error_text}
                }
    
    # Placeholder methods for other credential types
    async def _test_aws(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {'success': False, 'message': 'AWS credential testing not yet implemented'}
    
    async def _test_azure(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {'success': False, 'message': 'Azure credential testing not yet implemented'}
    
    async def _test_google_cloud(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {'success': False, 'message': 'Google Cloud credential testing not yet implemented'}
    
    async def _test_stripe(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {'success': False, 'message': 'Stripe credential testing not yet implemented'}
    
    async def _test_paypal(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {'success': False, 'message': 'PayPal credential testing not yet implemented'}
    
    async def _test_salesforce(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {'success': False, 'message': 'Salesforce credential testing not yet implemented'}
    
    async def _test_hubspot(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {'success': False, 'message': 'HubSpot credential testing not yet implemented'}
    
    async def _test_datadog(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {'success': False, 'message': 'Datadog credential testing not yet implemented'}
    
    async def _test_elasticsearch(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {'success': False, 'message': 'Elasticsearch credential testing not yet implemented'}
    
    async def _test_mongodb(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {'success': False, 'message': 'MongoDB credential testing not yet implemented'}
    
    async def _test_mysql(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {'success': False, 'message': 'MySQL credential testing not yet implemented'}
    
    async def _test_docker(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {'success': False, 'message': 'Docker credential testing not yet implemented'}
    
    async def _test_kubernetes(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {'success': False, 'message': 'Kubernetes credential testing not yet implemented'}
    
    async def _test_sendgrid(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {'success': False, 'message': 'SendGrid credential testing not yet implemented'}
    
    async def _test_twilio(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {'success': False, 'message': 'Twilio credential testing not yet implemented'}
    
    async def _test_gitlab(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {'success': False, 'message': 'GitLab credential testing not yet implemented'}
    
    async def _test_huggingface(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {'success': False, 'message': 'Hugging Face credential testing not yet implemented'}
    
    async def _test_oauth2_token(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {'success': False, 'message': 'OAuth2 token testing not yet implemented'}
    
    async def _test_http_basic_auth(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {'success': False, 'message': 'HTTP Basic Auth testing not yet implemented'}
    
    async def _test_s3(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {'success': False, 'message': 'S3 credential testing not yet implemented'}


# Usage example
async def test_credential_example():
    """Example of how to use the credential tester"""
    async with CredentialTester() as tester:
        # Test OpenAI credentials
        openai_data = {
            'api_key': 'sk-test-key',
            'base_url': 'https://api.openai.com/v1'
        }
        result = await tester.test_credential('openai_api', openai_data)
        print(f"OpenAI test result: {result}")
        
        # Test PostgreSQL credentials
        postgres_data = {
            'host': 'localhost',
            'port': 5432,
            'database': 'test_db',
            'user': 'test_user',
            'password': 'test_password'
        }
        result = await tester.test_credential('postgres_credentials', postgres_data)
        print(f"PostgreSQL test result: {result}")

if __name__ == "__main__":
    asyncio.run(test_credential_example())
