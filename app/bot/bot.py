"""Telegram bot initialization and management."""
import asyncio
from typing import Optional
from loguru import logger

from aiogram import Bot, Dispatcher

from app.core.settings import settings
from app.bot.handlers import register_handlers


class TelegramBot:
    """Telegram bot wrapper class."""
    
    def __init__(self):
        """Initialize bot and dispatcher."""
        self.bot: Optional[Bot] = None
        self.dp: Optional[Dispatcher] = None
        
    async def start(self):
        """Start the bot."""
        try:
            if not settings.telegram.bot_token:
                logger.warning("Telegram bot token not configured, skipping bot startup")
                return
                
            # Initialize bot
            self.bot = Bot(
                token=settings.telegram.bot_token
            )
            
            # Initialize dispatcher
            self.dp = Dispatcher()
            
            # Register handlers
            register_handlers(self.dp)
            logger.info("Telegram bot handlers registered")
            
            # Start polling
            await self.dp.start_polling(self.bot)
            logger.info("Telegram bot started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start Telegram bot: {e}")
            raise
    
    async def stop(self):
        """Stop the bot."""
        try:
            if self.bot:
                await self.bot.session.close()
                logger.info("Telegram bot stopped")
        except Exception as e:
            logger.error(f"Error stopping Telegram bot: {e}")


# Global bot instance
bot_instance = TelegramBot()


async def start_bot():
    """Start the Telegram bot."""
    await bot_instance.start()


async def stop_bot():
    """Stop the Telegram bot."""
    await bot_instance.stop()


def start_polling():
    """Start bot polling in a separate task."""
    try:
        if not settings.telegram.bot_token:
            logger.warning("Telegram bot token not configured, skipping bot startup")
            return
            
        # Initialize bot
        bot = Bot(
            token=settings.telegram.bot_token
        )
        
        # Initialize dispatcher
        dp = Dispatcher()
        
        # Register handlers
        register_handlers(dp)
        logger.info("Telegram bot handlers registered")
        
        # Start polling
        asyncio.create_task(dp.start_polling(bot))
        logger.info("Starting Telegram bot polling...")
        
    except Exception as e:
        logger.error(f"Failed to start Telegram bot: {e}")
        raise