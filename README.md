# RAG Telegram Bot

Production-ready RAG system with FastAPI, PostgreSQL, Elasticsearch, and Telegram Bot.

## Quick Start

1. **Clone and setup:**
   ```bash
   git clone <repository-url>
   cd RAG-with-Langchain-and-FastAPI
   cp env.example .env
   ```

2. **Configure environment variables in `.env`:**
   ```bash
   # Application
   APP_BASE_URL=http://localhost:8000
   
   # OpenAI
   AI__OPENAI_API_KEY=your_openai_api_key_here
   
   # Telegram Bot
   TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
   TELEGRAM__ADMIN_TELEGRAM_IDS=123456789,987654321
   ```

3. **Deploy with Docker:**
   ```bash
   docker-compose up -d --build
   ```

4. **Verify deployment:**
   - API Health: `http://localhost:8000/health`
   - Telegram Bot: Send `/start` to your bot

## API Endpoints

- `POST /api/v1/chat/query` - Chat with RAG system
- `GET /api/v1/docs/{doc_id}/download` - Download document
- `GET /health` - Health check

## Telegram Bot Commands

- `/start` - Welcome message
- `/upload` - Upload document (admin only)
- `/list` - List documents (admin only)
- `/delete <doc_id>` - Delete document (admin only)
- `/ask <question>` - Ask specific question

## Production Deployment

For external access, update `APP_BASE_URL` in `.env`:
```bash
APP_BASE_URL=http://your-server-ip:8000
```

Then restart:
```bash
docker-compose down
docker-compose up -d --build
```