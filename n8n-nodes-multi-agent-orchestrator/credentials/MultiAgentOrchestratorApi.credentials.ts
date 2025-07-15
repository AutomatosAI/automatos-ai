
import {
	IAuthenticateGeneric,
	ICredentialTestRequest,
	ICredentialType,
	INodeProperties,
} from 'n8n-workflow';

export class MultiAgentOrchestratorApi implements ICredentialType {
	name = 'multiAgentOrchestratorApi';
	displayName = 'Multi-Agent Orchestrator API';
	documentationUrl = 'https://docs.yourorg.com/multi-agent-orchestrator/authentication';
	properties: INodeProperties[] = [
		{
			displayName: 'API Base URL',
			name: 'baseUrl',
			type: 'string',
			default: 'http://localhost:8000',
			placeholder: 'https://your-orchestrator.example.com',
			description: 'Base URL of your Multi-Agent Orchestrator System',
			required: true,
		},
		{
			displayName: 'API Key',
			name: 'apiKey',
			type: 'string',
			typeOptions: {
				password: true,
			},
			default: '',
			placeholder: 'mao_1234567890abcdef...',
			description: 'API key for authentication with the Multi-Agent Orchestrator System',
			required: true,
		},
		{
			displayName: 'Timeout (seconds)',
			name: 'timeout',
			type: 'number',
			default: 300,
			description: 'Request timeout in seconds for long-running operations',
		},
	];

	authenticate: IAuthenticateGeneric = {
		type: 'generic',
		properties: {
			headers: {
				'Authorization': '=Bearer {{$credentials.apiKey}}',
				'Content-Type': 'application/json',
				'User-Agent': 'n8n-multi-agent-orchestrator/1.0.0',
			},
		},
	};

	test: ICredentialTestRequest = {
		request: {
			baseURL: '={{$credentials.baseUrl}}',
			url: '/api/v1/health',
			method: 'GET',
		},
		rules: [
			{
				type: 'responseSuccessBody',
				properties: {
					key: 'status',
					value: 'healthy',
				},
			},
		],
	};
}
