-- PostgreSQL Database Initialization Script
-- This script sets up the initial database structure for Automatos AI

-- Create database if it doesn't exist (this is handled by POSTGRES_DB env var)
-- CREATE DATABASE IF NOT EXISTS automatos_ai;

-- Set timezone
SET timezone = 'UTC';

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create initial schema (tables will be created by Alembic migrations)
-- This file ensures the database is ready for migrations

-- Log initialization
DO $$
BEGIN
    RAISE NOTICE 'PostgreSQL database initialized successfully for Automatos AI';
END $$;