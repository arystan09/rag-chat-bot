#!/bin/bash
set -e

echo "🚀 Starting RAG System..."

# Function to wait for PostgreSQL
wait_for_postgres() {
    echo "⏳ Waiting for PostgreSQL to be ready..."
    
    # Use environment variables from docker-compose
    DB_HOST=${DATABASE__HOST:-postgres}
    DB_PORT=${DATABASE__PORT:-5432}
    DB_USER=${DATABASE__USER:-raguser}
    
    echo "📡 Checking PostgreSQL at $DB_HOST:$DB_PORT..."
    
    # Wait for PostgreSQL to be ready
    until pg_isready -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER"; do
        echo "⏳ PostgreSQL is unavailable - sleeping for 2 seconds..."
        sleep 2
    done
    
    echo "✅ PostgreSQL is ready!"
}

# Function to wait for Elasticsearch
wait_for_elasticsearch() {
    echo "⏳ Waiting for Elasticsearch to be ready..."
    
    # Wait for Elasticsearch to be ready
    until curl -s http://elasticsearch:9200 >/dev/null; do
        echo "⏳ Waiting for Elasticsearch..."
        sleep 5
    done
    echo "✅ Elasticsearch is ready!"
}

# Function to run migrations
run_migrations() {
    echo "🔄 Running database migrations..."
    
    # Check for multiple heads and fix if necessary
    echo "🔍 Checking for migration conflicts..."
    HEADS_COUNT=$(/usr/local/bin/alembic heads | grep -v "INFO" | grep -v "\[" | wc -l)
    
    if [ "$HEADS_COUNT" -gt 1 ]; then
        echo "⚠️  Multiple migration heads detected, creating merge migration..."
        HEADS=$(/usr/local/bin/alembic heads | grep -v "INFO" | grep -v "\[" | tr '\n' ' ' | sed 's/ *$//')
        /usr/local/bin/alembic merge -m "merge heads" $HEADS
        echo "✅ Merge migration created"
    fi
    
    # Run migrations
    if /usr/local/bin/alembic upgrade head; then
        echo "✅ Database migrations completed successfully"
        return 0
    else
        echo "❌ Database migrations failed"
        return 1
    fi
}

# Function to start the application
start_app() {
    echo "🚀 Starting FastAPI application..."
    # Small delay to ensure all services are ready
    sleep 2
    exec uvicorn app.main:app --host 0.0.0.0 --port 8000
}

# Main execution
main() {
    # Wait for PostgreSQL
    wait_for_postgres
    
    # Wait for Elasticsearch
    wait_for_elasticsearch
    
    # Run migrations
    if ! run_migrations; then
        echo "⚠️  Migrations failed, but continuing with app startup..."
        echo "💡 You can run migrations manually with: make migrate"
    fi
    
    # Start the application
    start_app
}

# Run main function
main "$@"



