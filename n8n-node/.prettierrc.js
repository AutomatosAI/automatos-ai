
module.exports = {
	...require('prettier-config-n8n'),
	overrides: [
		{
			files: ['*.json'],
			options: {
				tabWidth: 2,
			},
		},
	],
};
