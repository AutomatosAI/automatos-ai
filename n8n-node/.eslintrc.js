
module.exports = {
	root: true,
	env: {
		browser: true,
		es6: true,
		node: true,
	},
	parser: '@typescript-eslint/parser',
	parserOptions: {
		project: 'tsconfig.json',
		sourceType: 'module',
		extraFileExtensions: ['.json'],
	},
	plugins: ['@typescript-eslint', 'n8n-nodes-base'],
	extends: [
		'eslint:recommended',
		'@typescript-eslint/recommended',
		'plugin:n8n-nodes-base/nodes',
	],
	rules: {
		'n8n-nodes-base/node-param-default-missing': 'error',
		'n8n-nodes-base/node-param-description-missing': 'error',
		'n8n-nodes-base/node-param-display-name-miscased': 'error',
		'n8n-nodes-base/node-param-display-name-not-first-position': 'error',
		'n8n-nodes-base/node-param-name-miscased': 'error',
		'n8n-nodes-base/node-param-option-name-duplicate': 'error',
		'n8n-nodes-base/node-param-option-name-wrong-for-get-many': 'error',
		'n8n-nodes-base/node-param-option-value-duplicate': 'error',
		'n8n-nodes-base/node-param-required-false': 'error',
		'n8n-nodes-base/node-param-type-options-missing-from-limit': 'error',
		'@typescript-eslint/no-unused-vars': ['error', { argsIgnorePattern: '^_' }],
		'@typescript-eslint/explicit-function-return-type': 'off',
		'@typescript-eslint/no-explicit-any': 'off',
	},
	ignorePatterns: ['dist/**', 'node_modules/**'],
};
