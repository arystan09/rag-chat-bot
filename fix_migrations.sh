#!/bin/bash

# Script to fix Alembic migration conflicts
# Usage: ./fix_migrations.sh

echo "🔧 Fixing Alembic migration conflicts..."

# Check if there are multiple heads
echo "📋 Checking current migration heads..."
docker exec rag_app alembic heads

# Get the heads
HEADS=$(docker exec rag_app alembic heads | grep -v "INFO" | tr '\n' ' ' | sed 's/ *$//')

if [ $(echo $HEADS | wc -w) -gt 1 ]; then
    echo "⚠️  Multiple heads detected: $HEADS"
    echo "🔄 Creating merge migration..."
    
    # Create merge migration
    docker exec rag_app alembic merge -m "merge heads" $HEADS
    
    echo "📈 Applying merge migration..."
    docker exec rag_app alembic upgrade head
    
    echo "✅ Migration conflicts resolved!"
else
    echo "✅ No migration conflicts detected."
fi

echo "📊 Current migration status:"
docker exec rag_app alembic current


