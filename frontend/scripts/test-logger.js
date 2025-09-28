/**
 * Test script for the logger implementation
 */
const logger = require('../lib/logger');

console.log('Starting logger test...');

// Test different log levels
logger.debug('This is a debug message', { details: 'Some debug details' });
logger.info('This is an info message', { user: 'test-user', action: 'login' });
logger.warn('This is a warning message', { type: 'performance', threshold: 500 });
logger.error('This is an error message', new Error('Test error'));

// Test with different data types
logger.info('String test:', 'Simple string');
logger.info('Number test:', 123.45);
logger.info('Boolean test:', true);
logger.info('Array test:', [1, 2, 3, 'test']);
logger.info('Object test:', { name: 'Test Object', properties: { a: 1, b: 2 } });
logger.info('Null test:', null);
logger.info('Undefined test:', undefined);

// Test with multiple arguments
logger.info('Multiple arguments:', 'arg1', 'arg2', { key: 'value' }, [1, 2, 3]);

console.log('Logger test completed!');
