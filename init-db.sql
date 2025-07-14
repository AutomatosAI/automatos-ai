CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS embeddings (
    id SERIAL PRIMARY KEY,
    embedding VECTOR(384),  -- Dimension for MiniLM; adjust if needed
    context TEXT
);
-- Optional mock insert for testing
INSERT INTO embeddings (embedding, context) VALUES ('[0.1,0.2,0.3,...,0.384]', 'Sample context: Use FastAPI for secure APIs.');