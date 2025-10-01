#!/bin/bash
# Apply document_usage tracking migration to production database
# Run this script to add the document_usage table for Context Engineering

set -e  # Exit on error

echo "===================================================="
echo "Document Usage Tracking Migration"
echo "===================================================="
echo ""

# Database connection details
DB_HOST="206.81.0.227"
DB_USER="postgres"
DB_NAME="automatos"
SSH_KEY="/Users/gkavanagh/.ssh/id_rsa"

echo "📊 Checking current database state..."
echo ""

# Check if table already exists
TABLE_EXISTS=$(ssh -i $SSH_KEY root@$DB_HOST "psql -U $DB_USER -d $DB_NAME -tAc \"SELECT COUNT(*) FROM information_schema.tables WHERE table_name='document_usage'\"")

if [ "$TABLE_EXISTS" -gt "0" ]; then
    echo "⚠️  WARNING: document_usage table already exists!"
    echo ""
    read -p "Do you want to drop and recreate it? (yes/no): " -r
    echo ""
    if [[ $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        echo "🗑️  Dropping existing table..."
        ssh -i $SSH_KEY root@$DB_HOST "psql -U $DB_USER -d $DB_NAME -c 'DROP TABLE IF EXISTS document_usage CASCADE'"
        echo "✅ Table dropped"
        echo ""
    else
        echo "❌ Migration cancelled"
        exit 1
    fi
fi

echo "📝 Applying migration..."
echo ""

# Apply the migration from init_complete_schema.sql
ssh -i $SSH_KEY root@$DB_HOST "psql -U $DB_USER -d $DB_NAME" < init_complete_schema.sql

echo "✅ Migration applied successfully!"
echo ""

echo "🔍 Verifying table structure..."
echo ""
ssh -i $SSH_KEY root@$DB_HOST "psql -U $DB_USER -d $DB_NAME -c '\\d document_usage'"

echo ""
echo "📊 Current row count:"
ssh -i $SSH_KEY root@$DB_HOST "psql -U $DB_USER -d $DB_NAME -tAc 'SELECT COUNT(*) FROM document_usage'"

echo ""
echo "===================================================="
echo "✅ Migration Complete!"
echo "===================================================="
echo ""
echo "Next steps:"
echo "1. Test the tracking by running: python generate_test_usage_data.py"
echo "2. Verify the Context Engineering page shows real data"
echo "3. Check the logs for any insert errors"
echo ""

