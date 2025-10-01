-- Insert test data directly into document_usage table
-- This creates realistic tracking data for testing Context Engineering page

-- Insert document search events (last 24 hours)
INSERT INTO document_usage (event_type, query, results_count, execution_time_ms, metadata, timestamp) VALUES
('document_searched', 'How do I configure authentication?', 8, 245, '{"min_similarity": 0.7}', NOW() - INTERVAL '30 minutes'),
('document_searched', 'What is the deployment process?', 12, 189, '{"min_similarity": 0.7}', NOW() - INTERVAL '1 hour'),
('document_searched', 'Database optimization strategies', 5, 422, '{"min_similarity": 0.7}', NOW() - INTERVAL '2 hours'),
('document_searched', 'Error handling best practices', 9, 312, '{"min_similarity": 0.7}', NOW() - INTERVAL '3 hours'),
('document_searched', 'API rate limiting implementation', 7, 278, '{"min_similarity": 0.7}', NOW() - INTERVAL '4 hours'),
('document_searched', 'Caching strategies for performance', 11, 167, '{"min_similarity": 0.7}', NOW() - INTERVAL '5 hours'),
('document_searched', 'Security best practices guide', 14, 134, '{"min_similarity": 0.7}', NOW() - INTERVAL '6 hours'),
('document_searched', 'Microservices architecture patterns', 6, 398, '{"min_similarity": 0.7}', NOW() - INTERVAL '7 hours'),
('document_searched', 'Docker container optimization', 10, 223, '{"min_similarity": 0.7}', NOW() - INTERVAL '8 hours'),
('document_searched', 'CI/CD pipeline setup', 8, 267, '{"min_similarity": 0.7}', NOW() - INTERVAL '9 hours'),

-- Insert RAG query events
('rag_query', 'Explain authentication flow', 5, 456, '{"max_chunks": 5, "max_tokens": 2000}', NOW() - INTERVAL '45 minutes'),
('rag_query', 'How does the deployment work?', 7, 523, '{"max_chunks": 5, "max_tokens": 2000}', NOW() - INTERVAL '1.5 hours'),
('rag_query', 'Best practices for error handling', 6, 489, '{"max_chunks": 5, "max_tokens": 2000}', NOW() - INTERVAL '2.5 hours'),
('rag_query', 'Rate limiting strategies', 4, 412, '{"max_chunks": 5, "max_tokens": 2000}', NOW() - INTERVAL '3.5 hours'),
('rag_query', 'Security implementation guide', 8, 567, '{"max_chunks": 5, "max_tokens": 2000}', NOW() - INTERVAL '5.5 hours'),
('rag_query', 'Microservices patterns', 5, 445, '{"max_chunks": 5, "max_tokens": 2000}', NOW() - INTERVAL '7.5 hours'),

-- Insert some failed queries (0 results)
('document_searched', 'nonexistent query xyz123', 0, 145, '{"min_similarity": 0.7}', NOW() - INTERVAL '1.2 hours'),
('document_searched', 'random test query', 0, 123, '{"min_similarity": 0.7}', NOW() - INTERVAL '4.3 hours'),

-- Insert older events for time-series data
('document_searched', 'legacy system migration', 9, 234, '{"min_similarity": 0.7}', NOW() - INTERVAL '12 hours'),
('document_searched', 'kubernetes deployment', 11, 345, '{"min_similarity": 0.7}', NOW() - INTERVAL '14 hours'),
('rag_query', 'explain monitoring setup', 6, 478, '{"max_chunks": 5, "max_tokens": 2000}', NOW() - INTERVAL '16 hours'),
('document_searched', 'logging best practices', 7, 256, '{"min_similarity": 0.7}', NOW() - INTERVAL '18 hours'),
('rag_query', 'database scaling strategies', 5, 523, '{"max_chunks": 5, "max_tokens": 2000}', NOW() - INTERVAL '20 hours'),
('document_searched', 'API versioning guide', 8, 198, '{"min_similarity": 0.7}', NOW() - INTERVAL '22 hours');

-- Verify the inserts
SELECT 'Inserted test data successfully!' as status;
SELECT event_type, COUNT(*) as count, AVG(execution_time_ms) as avg_time FROM document_usage GROUP BY event_type;
SELECT COUNT(*) as total_events FROM document_usage;

