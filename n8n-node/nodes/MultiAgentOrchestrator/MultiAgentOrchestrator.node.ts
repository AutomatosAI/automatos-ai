
import {
	IExecuteFunctions,
	INodeExecutionData,
	INodeType,
	INodeTypeDescription,
	NodeOperationError,
	NodeApiError,
} from 'n8n-workflow';

export class MultiAgentOrchestrator implements INodeType {
	description: INodeTypeDescription = {
		displayName: 'Multi-Agent Orchestrator',
		name: 'multiAgentOrchestrator',
		icon: 'file:multiagent.svg',
		group: ['transform'],
		version: 1,
		subtitle: '={{$parameter["operation"] + ": " + $parameter["resource"]}}',
		description: 'Interact with Multi-Agent Orchestration System for workflow automation, document management, and context engineering',
		defaults: {
			name: 'Multi-Agent Orchestrator',
		},
		inputs: ['main'],
		outputs: ['main'],
		credentials: [
			{
				name: 'multiAgentOrchestratorApi',
				required: true,
			},
		],
		requestDefaults: {
			baseURL: '={{$credentials.baseUrl}}',
			headers: {
				Accept: 'application/json',
				'Content-Type': 'application/json',
			},
		},
		properties: [
			{
				displayName: 'Resource',
				name: 'resource',
				type: 'options',
				noDataExpression: true,
				options: [
					{
						name: 'Workflow',
						value: 'workflow',
					},
					{
						name: 'Document',
						value: 'document',
					},
					{
						name: 'Context',
						value: 'context',
					},
					{
						name: 'Agent',
						value: 'agent',
					},
				],
				default: 'workflow',
			},

			// Workflow Operations
			{
				displayName: 'Operation',
				name: 'operation',
				type: 'options',
				noDataExpression: true,
				displayOptions: {
					show: {
						resource: ['workflow'],
					},
				},
				options: [
					{
						name: 'Start',
						value: 'start',
						description: 'Start a new workflow execution',
						action: 'Start a workflow',
					},
					{
						name: 'Get Status',
						value: 'getStatus',
						description: 'Get the status of a workflow execution',
						action: 'Get workflow status',
					},
					{
						name: 'Stop',
						value: 'stop',
						description: 'Stop a running workflow execution',
						action: 'Stop a workflow',
					},
					{
						name: 'List',
						value: 'list',
						description: 'List all workflow executions',
						action: 'List workflows',
					},
				],
				default: 'start',
			},

			// Document Operations
			{
				displayName: 'Operation',
				name: 'operation',
				type: 'options',
				noDataExpression: true,
				displayOptions: {
					show: {
						resource: ['document'],
					},
				},
				options: [
					{
						name: 'Upload',
						value: 'upload',
						description: 'Upload a document to the system',
						action: 'Upload a document',
					},
					{
						name: 'Get',
						value: 'get',
						description: 'Get document information',
						action: 'Get a document',
					},
					{
						name: 'Delete',
						value: 'delete',
						description: 'Delete a document',
						action: 'Delete a document',
					},
					{
						name: 'List',
						value: 'list',
						description: 'List all documents',
						action: 'List documents',
					},
				],
				default: 'upload',
			},

			// Context Operations
			{
				displayName: 'Operation',
				name: 'operation',
				type: 'options',
				noDataExpression: true,
				displayOptions: {
					show: {
						resource: ['context'],
					},
				},
				options: [
					{
						name: 'Search',
						value: 'search',
						description: 'Search through context using semantic search',
						action: 'Search context',
					},
					{
						name: 'Add',
						value: 'add',
						description: 'Add new context information',
						action: 'Add context',
					},
					{
						name: 'Update',
						value: 'update',
						description: 'Update existing context',
						action: 'Update context',
					},
					{
						name: 'Delete',
						value: 'delete',
						description: 'Delete context information',
						action: 'Delete context',
					},
				],
				default: 'search',
			},

			// Agent Operations
			{
				displayName: 'Operation',
				name: 'operation',
				type: 'options',
				noDataExpression: true,
				displayOptions: {
					show: {
						resource: ['agent'],
					},
				},
				options: [
					{
						name: 'List',
						value: 'list',
						description: 'List available agents',
						action: 'List agents',
					},
					{
						name: 'Get Status',
						value: 'getStatus',
						description: 'Get agent status and capabilities',
						action: 'Get agent status',
					},
					{
						name: 'Execute Task',
						value: 'executeTask',
						description: 'Execute a task using a specific agent',
						action: 'Execute agent task',
					},
				],
				default: 'list',
			},

			// Workflow Parameters
			{
				displayName: 'Workflow ID',
				name: 'workflowId',
				type: 'string',
				displayOptions: {
					show: {
						resource: ['workflow'],
						operation: ['getStatus', 'stop'],
					},
				},
				default: '',
				placeholder: 'wf_1234567890',
				description: 'The ID of the workflow execution',
				required: true,
			},
			{
				displayName: 'Workflow Template',
				name: 'workflowTemplate',
				type: 'string',
				displayOptions: {
					show: {
						resource: ['workflow'],
						operation: ['start'],
					},
				},
				default: '',
				placeholder: 'data-analysis-pipeline',
				description: 'Name or ID of the workflow template to execute',
				required: true,
			},
			{
				displayName: 'Input Parameters',
				name: 'inputParameters',
				type: 'json',
				displayOptions: {
					show: {
						resource: ['workflow'],
						operation: ['start'],
					},
				},
				default: '{}',
				description: 'JSON object containing input parameters for the workflow',
			},
			{
				displayName: 'Priority',
				name: 'priority',
				type: 'options',
				displayOptions: {
					show: {
						resource: ['workflow'],
						operation: ['start'],
					},
				},
				options: [
					{
						name: 'Low',
						value: 'low',
					},
					{
						name: 'Normal',
						value: 'normal',
					},
					{
						name: 'High',
						value: 'high',
					},
					{
						name: 'Critical',
						value: 'critical',
					},
				],
				default: 'normal',
				description: 'Priority level for workflow execution',
			},

			// Document Parameters
			{
				displayName: 'Document ID',
				name: 'documentId',
				type: 'string',
				displayOptions: {
					show: {
						resource: ['document'],
						operation: ['get', 'delete'],
					},
				},
				default: '',
				placeholder: 'doc_1234567890',
				description: 'The ID of the document',
				required: true,
			},
			{
				displayName: 'File Content',
				name: 'fileContent',
				type: 'string',
				displayOptions: {
					show: {
						resource: ['document'],
						operation: ['upload'],
					},
				},
				default: '',
				description: 'Content of the file to upload (base64 encoded for binary files)',
				required: true,
			},
			{
				displayName: 'File Name',
				name: 'fileName',
				type: 'string',
				displayOptions: {
					show: {
						resource: ['document'],
						operation: ['upload'],
					},
				},
				default: '',
				placeholder: 'document.pdf',
				description: 'Name of the file being uploaded',
				required: true,
			},
			{
				displayName: 'File Type',
				name: 'fileType',
				type: 'options',
				displayOptions: {
					show: {
						resource: ['document'],
						operation: ['upload'],
					},
				},
				options: [
					{
						name: 'PDF',
						value: 'pdf',
					},
					{
						name: 'Text',
						value: 'txt',
					},
					{
						name: 'Word Document',
						value: 'docx',
					},
					{
						name: 'Markdown',
						value: 'md',
					},
					{
						name: 'JSON',
						value: 'json',
					},
					{
						name: 'CSV',
						value: 'csv',
					},
				],
				default: 'pdf',
				description: 'Type of the document being uploaded',
			},
			{
				displayName: 'Tags',
				name: 'tags',
				type: 'string',
				displayOptions: {
					show: {
						resource: ['document'],
						operation: ['upload'],
					},
				},
				default: '',
				placeholder: 'research, analysis, quarterly-report',
				description: 'Comma-separated tags for document categorization',
			},

			// Context Parameters
			{
				displayName: 'Search Query',
				name: 'searchQuery',
				type: 'string',
				displayOptions: {
					show: {
						resource: ['context'],
						operation: ['search'],
					},
				},
				default: '',
				placeholder: 'Find information about quarterly sales performance',
				description: 'Natural language query for semantic search',
				required: true,
			},
			{
				displayName: 'Max Results',
				name: 'maxResults',
				type: 'number',
				displayOptions: {
					show: {
						resource: ['context'],
						operation: ['search'],
					},
				},
				default: 10,
				description: 'Maximum number of search results to return',
			},
			{
				displayName: 'Context ID',
				name: 'contextId',
				type: 'string',
				displayOptions: {
					show: {
						resource: ['context'],
						operation: ['update', 'delete'],
					},
				},
				default: '',
				placeholder: 'ctx_1234567890',
				description: 'The ID of the context entry',
				required: true,
			},
			{
				displayName: 'Context Content',
				name: 'contextContent',
				type: 'string',
				displayOptions: {
					show: {
						resource: ['context'],
						operation: ['add', 'update'],
					},
				},
				default: '',
				description: 'The content to add or update in the context',
				required: true,
			},
			{
				displayName: 'Context Type',
				name: 'contextType',
				type: 'options',
				displayOptions: {
					show: {
						resource: ['context'],
						operation: ['add'],
					},
				},
				options: [
					{
						name: 'Knowledge',
						value: 'knowledge',
					},
					{
						name: 'Procedure',
						value: 'procedure',
					},
					{
						name: 'Template',
						value: 'template',
					},
					{
						name: 'Reference',
						value: 'reference',
					},
				],
				default: 'knowledge',
				description: 'Type of context being added',
			},

			// Agent Parameters
			{
				displayName: 'Agent ID',
				name: 'agentId',
				type: 'string',
				displayOptions: {
					show: {
						resource: ['agent'],
						operation: ['getStatus', 'executeTask'],
					},
				},
				default: '',
				placeholder: 'agent_data_analyst',
				description: 'The ID of the agent',
				required: true,
			},
			{
				displayName: 'Task Description',
				name: 'taskDescription',
				type: 'string',
				displayOptions: {
					show: {
						resource: ['agent'],
						operation: ['executeTask'],
					},
				},
				default: '',
				placeholder: 'Analyze the uploaded dataset and provide insights',
				description: 'Description of the task for the agent to execute',
				required: true,
			},
			{
				displayName: 'Task Parameters',
				name: 'taskParameters',
				type: 'json',
				displayOptions: {
					show: {
						resource: ['agent'],
						operation: ['executeTask'],
					},
				},
				default: '{}',
				description: 'JSON object containing parameters for the agent task',
			},

			// Common Parameters
			{
				displayName: 'Additional Options',
				name: 'additionalOptions',
				type: 'collection',
				placeholder: 'Add Option',
				default: {},
				options: [
					{
						displayName: 'Timeout (seconds)',
						name: 'timeout',
						type: 'number',
						default: 300,
						description: 'Request timeout in seconds',
					},
					{
						displayName: 'Async Execution',
						name: 'async',
						type: 'boolean',
						default: false,
						description: 'Whether to execute the operation asynchronously',
					},
					{
						displayName: 'Callback URL',
						name: 'callbackUrl',
						type: 'string',
						default: '',
						placeholder: 'https://your-webhook.example.com/callback',
						description: 'URL to receive callback notifications for async operations',
					},
				],
			},
		],
	};

	async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
		const items = this.getInputData();
		const returnData: INodeExecutionData[] = [];

		for (let i = 0; i < items.length; i++) {
			const resource = this.getNodeParameter('resource', i) as string;
			const operation = this.getNodeParameter('operation', i) as string;

			try {
				let responseData: any;

				if (resource === 'workflow') {
					responseData = await this.handleWorkflowOperations(operation, i);
				} else if (resource === 'document') {
					responseData = await this.handleDocumentOperations(operation, i);
				} else if (resource === 'context') {
					responseData = await this.handleContextOperations(operation, i);
				} else if (resource === 'agent') {
					responseData = await this.handleAgentOperations(operation, i);
				} else {
					throw new NodeOperationError(this.getNode(), `Unknown resource: ${resource}`, {
						itemIndex: i,
					});
				}

				const executionData = this.helpers.constructExecutionMetaData(
					this.helpers.returnJsonArray(responseData),
					{ itemData: { item: i } },
				);

				returnData.push(...executionData);
			} catch (error) {
				if (this.continueOnFail()) {
					const executionData = this.helpers.constructExecutionMetaData(
						this.helpers.returnJsonArray({ error: error.message }),
						{ itemData: { item: i } },
					);
					returnData.push(...executionData);
					continue;
				}
				throw error;
			}
		}

		return [returnData];
	}

	private async handleWorkflowOperations(operation: string, itemIndex: number): Promise<any> {
		const additionalOptions = this.getNodeParameter('additionalOptions', itemIndex) as any;

		switch (operation) {
			case 'start':
				const workflowTemplate = this.getNodeParameter('workflowTemplate', itemIndex) as string;
				const inputParameters = this.getNodeParameter('inputParameters', itemIndex) as string;
				const priority = this.getNodeParameter('priority', itemIndex) as string;

				const startPayload: any = {
					template: workflowTemplate,
					priority,
					parameters: JSON.parse(inputParameters || '{}'),
				};

				if (additionalOptions.async) {
					startPayload.async = true;
					if (additionalOptions.callbackUrl) {
						startPayload.callbackUrl = additionalOptions.callbackUrl;
					}
				}

				return await this.makeApiRequest('POST', '/api/v1/workflows', startPayload, itemIndex);

			case 'getStatus':
				const workflowId = this.getNodeParameter('workflowId', itemIndex) as string;
				return await this.makeApiRequest('GET', `/api/v1/workflows/${workflowId}`, undefined, itemIndex);

			case 'stop':
				const stopWorkflowId = this.getNodeParameter('workflowId', itemIndex) as string;
				return await this.makeApiRequest('POST', `/api/v1/workflows/${stopWorkflowId}/stop`, undefined, itemIndex);

			case 'list':
				return await this.makeApiRequest('GET', '/api/v1/workflows', undefined, itemIndex);

			default:
				throw new NodeOperationError(this.getNode(), `Unknown workflow operation: ${operation}`, {
					itemIndex,
				});
		}
	}

	private async handleDocumentOperations(operation: string, itemIndex: number): Promise<any> {
		switch (operation) {
			case 'upload':
				const fileContent = this.getNodeParameter('fileContent', itemIndex) as string;
				const fileName = this.getNodeParameter('fileName', itemIndex) as string;
				const fileType = this.getNodeParameter('fileType', itemIndex) as string;
				const tags = this.getNodeParameter('tags', itemIndex) as string;

				const uploadPayload = {
					content: fileContent,
					filename: fileName,
					type: fileType,
					tags: tags ? tags.split(',').map(tag => tag.trim()) : [],
				};

				return await this.makeApiRequest('POST', '/api/v1/documents', uploadPayload, itemIndex);

			case 'get':
				const documentId = this.getNodeParameter('documentId', itemIndex) as string;
				return await this.makeApiRequest('GET', `/api/v1/documents/${documentId}`, undefined, itemIndex);

			case 'delete':
				const deleteDocumentId = this.getNodeParameter('documentId', itemIndex) as string;
				return await this.makeApiRequest('DELETE', `/api/v1/documents/${deleteDocumentId}`, undefined, itemIndex);

			case 'list':
				return await this.makeApiRequest('GET', '/api/v1/documents', undefined, itemIndex);

			default:
				throw new NodeOperationError(this.getNode(), `Unknown document operation: ${operation}`, {
					itemIndex,
				});
		}
	}

	private async handleContextOperations(operation: string, itemIndex: number): Promise<any> {
		switch (operation) {
			case 'search':
				const searchQuery = this.getNodeParameter('searchQuery', itemIndex) as string;
				const maxResults = this.getNodeParameter('maxResults', itemIndex) as number;

				const searchPayload = {
					query: searchQuery,
					maxResults,
				};

				return await this.makeApiRequest('POST', '/api/v1/context/search', searchPayload, itemIndex);

			case 'add':
				const contextContent = this.getNodeParameter('contextContent', itemIndex) as string;
				const contextType = this.getNodeParameter('contextType', itemIndex) as string;

				const addPayload = {
					content: contextContent,
					type: contextType,
				};

				return await this.makeApiRequest('POST', '/api/v1/context', addPayload, itemIndex);

			case 'update':
				const updateContextId = this.getNodeParameter('contextId', itemIndex) as string;
				const updateContextContent = this.getNodeParameter('contextContent', itemIndex) as string;

				const updatePayload = {
					content: updateContextContent,
				};

				return await this.makeApiRequest('PUT', `/api/v1/context/${updateContextId}`, updatePayload, itemIndex);

			case 'delete':
				const deleteContextId = this.getNodeParameter('contextId', itemIndex) as string;
				return await this.makeApiRequest('DELETE', `/api/v1/context/${deleteContextId}`, undefined, itemIndex);

			default:
				throw new NodeOperationError(this.getNode(), `Unknown context operation: ${operation}`, {
					itemIndex,
				});
		}
	}

	private async handleAgentOperations(operation: string, itemIndex: number): Promise<any> {
		switch (operation) {
			case 'list':
				return await this.makeApiRequest('GET', '/api/v1/agents', undefined, itemIndex);

			case 'getStatus':
				const agentId = this.getNodeParameter('agentId', itemIndex) as string;
				return await this.makeApiRequest('GET', `/api/v1/agents/${agentId}`, undefined, itemIndex);

			case 'executeTask':
				const taskAgentId = this.getNodeParameter('agentId', itemIndex) as string;
				const taskDescription = this.getNodeParameter('taskDescription', itemIndex) as string;
				const taskParameters = this.getNodeParameter('taskParameters', itemIndex) as string;

				const taskPayload = {
					description: taskDescription,
					parameters: JSON.parse(taskParameters || '{}'),
				};

				return await this.makeApiRequest('POST', `/api/v1/agents/${taskAgentId}/execute`, taskPayload, itemIndex);

			default:
				throw new NodeOperationError(this.getNode(), `Unknown agent operation: ${operation}`, {
					itemIndex,
				});
		}
	}

	private async makeApiRequest(method: string, endpoint: string, body?: any, itemIndex?: number): Promise<any> {
		const additionalOptions = this.getNodeParameter('additionalOptions', itemIndex || 0) as any;
		const timeout = additionalOptions.timeout || 300;

		const options = {
			method,
			url: endpoint,
			body,
			json: true,
			timeout: timeout * 1000,
		};

		try {
			const response = await this.helpers.httpRequestWithAuthentication.call(
				this,
				'multiAgentOrchestratorApi',
				options,
			);
			return response;
		} catch (error) {
			if (error.response?.status === 401) {
				throw new NodeApiError(this.getNode(), error, {
					message: 'Invalid API credentials',
					description: 'Please check your API key and base URL',
				});
			} else if (error.response?.status === 404) {
				throw new NodeApiError(this.getNode(), error, {
					message: 'Resource not found',
					description: 'The requested resource does not exist',
				});
			} else if (error.response?.status >= 500) {
				throw new NodeApiError(this.getNode(), error, {
					message: 'Server error',
					description: 'The Multi-Agent Orchestrator System is experiencing issues',
				});
			}
			throw new NodeApiError(this.getNode(), error);
		}
	}
}
