#!/bin/bash

# Wait for database to be ready
echo "Waiting for database..."
while ! pg_isready -h postgres -p 5432 -U postgres; do
  sleep 1
done
echo "Database is ready!"

# Run Alembic migrations
echo "Running database migrations..."
alembic upgrade head

# Start the application
echo "Starting Automatos AI backend..."
exec python main.py

