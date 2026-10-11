#!/bin/sh
# Start the image's local PostgreSQL and make sure the test role and database exist.
# Used by .oss-scanner/Dockerfile and by anyone reproducing a finding in the image.
set -e
service postgresql start
until pg_isready -h 127.0.0.1 -p 5432 >/dev/null 2>&1; do sleep 1; done
su postgres -c "psql -tAc \"SELECT 1 FROM pg_roles WHERE rolname = 'test'\"" | grep -q 1 \
  || su postgres -c "psql -c \"CREATE ROLE test LOGIN PASSWORD 'test'\""
su postgres -c "psql -tAc \"SELECT 1 FROM pg_database WHERE datname = 'test_db'\"" | grep -q 1 \
  || su postgres -c "createdb -O test test_db"
