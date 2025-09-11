"""Telegram bot handlers."""
import asyncio
import tempfile
import os
import uuid
from typing import Optional, Dict
from aiogram import Router, F
from aiogram.types import Message, CallbackQuery
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from loguru import logger

from app.db.session import SessionLocal
from app.db.models import User
from app.ingestion.indexer import index_document, delete_document, get_user_documents
from app.api.schemas import QueryRequest
from app.core.settings import settings


router = Router()

# Store active conversations per user
user_conversations: Dict[int, str] = {}


async def check_user_role(telegram_id: int) -> Optional[str]:
    """
    Check user role by Telegram ID.
    
    Args:
        telegram_id: Telegram user ID
        
    Returns:
        User role ('admin' or 'user') or None if user not found
    """
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(telegram_id)).first()
        return user.role if user else None
    finally:
        db.close()


async def require_admin(message: Message) -> bool:
    """
    Check if user has admin role.
    
    Args:
        message: Telegram message
        
    Returns:
        True if user is admin, False otherwise
    """
    role = await check_user_role(message.from_user.id)
    if role != 'admin':
        await message.reply(
            "❌ У вас нет прав для этой команды. Обратитесь к администратору.",
            disable_web_page_preview=True
        )
        logger.warning(f"Non-admin user {message.from_user.id} attempted admin command")
        return False
    return True


async def require_super_admin(message: Message) -> bool:
    """
    Check if user is in the initial admin list from settings.
    
    Args:
        message: Telegram message
        
    Returns:
        True if user is super admin, False otherwise
    """
    telegram_id = str(message.from_user.id)
    if telegram_id not in settings.telegram.admin_telegram_ids:
        await message.reply(
            "❌ У вас нет прав супер-администратора для этой команды.",
            disable_web_page_preview=True
        )
        logger.warning(f"Non-super-admin user {message.from_user.id} attempted super admin command")
        return False
    return True


async def add_admin_to_db(telegram_id: int) -> bool:
    """
    Add user as admin to database.
    
    Args:
        telegram_id: Telegram user ID
        
    Returns:
        True if successful, False otherwise
    """
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(telegram_id)).first()
        if user:
            user.role = 'admin'
            db.commit()
            logger.info(f"Updated user {telegram_id} to admin role")
            return True
        else:
            # Create new admin user
            user = User(
                telegram_id=str(telegram_id),
                role='admin'
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            logger.info(f"Created new admin user {telegram_id}")
            return True
    except Exception as e:
        logger.error(f"Failed to add admin {telegram_id}: {e}")
        db.rollback()
        return False
    finally:
        db.close()


async def remove_admin_from_db(telegram_id: int) -> bool:
    """
    Remove admin role from user in database.
    
    Args:
        telegram_id: Telegram user ID
        
    Returns:
        True if successful, False otherwise
    """
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(telegram_id)).first()
        if user:
            user.role = 'user'
            db.commit()
            logger.info(f"Removed admin role from user {telegram_id}")
            return True
        return False
    except Exception as e:
        logger.error(f"Failed to remove admin {telegram_id}: {e}")
        db.rollback()
        return False
    finally:
        db.close()


async def get_all_admins() -> list:
    """
    Get list of all admin users.
    
    Returns:
        List of admin user info
    """
    db = SessionLocal()
    try:
        admins = db.query(User).filter(User.role == 'admin').all()
        return [
            {
                'telegram_id': admin.telegram_id,
                'role': admin.role,
                'created_at': admin.created_at.isoformat() if admin.created_at else None
            }
            for admin in admins
        ]
    except Exception as e:
        logger.error(f"Failed to get admins: {e}")
        return []
    finally:
        db.close()


async def send_images_from_chunks(message: Message, retrieved_chunks: list) -> None:
    """
    Send images from retrieved chunks with limits.
    
    Args:
        message: Telegram message to reply to
        retrieved_chunks: List of retrieved chunks with image_urls
    """
    try:
        # Collect all image URLs from chunks
        all_image_urls = []
        for chunk in retrieved_chunks:
            if chunk.get('has_image') and chunk.get('image_urls'):
                # Limit images per chunk
                chunk_images = chunk['image_urls'][:settings.telegram.max_images_per_chunk]
                all_image_urls.extend(chunk_images)
        
        # Deduplicate and limit total images
        unique_images = list(dict.fromkeys(all_image_urls))  # Preserve order, remove duplicates
        images_to_send = unique_images[:settings.telegram.max_images_per_response]
        
        if images_to_send:
            logger.info(f"Sending {len(images_to_send)} images from retrieved chunks")
            
            # Send images one by one
            for image_url in images_to_send:
                try:
                    await message.bot.send_photo(
                        chat_id=message.chat.id,
                        photo=image_url,
                        caption=f"📷 Изображение из документа"
                    )
                except Exception as e:
                    logger.warning(f"Failed to send image {image_url}: {e}")
                    continue
                    
    except Exception as e:
        logger.error(f"Error sending images from chunks: {e}")


async def get_or_create_user(telegram_id: int) -> Optional[User]:
    """
    Get or create user from Telegram ID.
    
    Args:
        telegram_id: Telegram user ID
        
    Returns:
        User object or None if failed
    """
    db = SessionLocal()
    try:
        # Check if user exists
        user = db.query(User).filter(User.telegram_id == str(telegram_id)).first()
        
        # Determine current role based on admin settings
        current_role = 'admin' if str(telegram_id) in settings.telegram.admin_telegram_ids else 'user'
        
        if not user:
            # Create new user
            user = User(
                telegram_id=str(telegram_id),
                role=current_role
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            logger.info(f"Created new user with Telegram ID {telegram_id}, role: {current_role}")
        else:
            # Update existing user's role if it changed
            if user.role != current_role:
                user.role = current_role
                db.commit()
                logger.info(f"Updated user {telegram_id} role from {user.role} to {current_role}")
            else:
                logger.info(f"Found existing user with Telegram ID {telegram_id}, role: {user.role}")
        
        return user
        
    except Exception as e:
        logger.error(f"Failed to get/create user: {e}")
        db.rollback()
        return None
    finally:
        db.close()


def get_user_conversation_id(telegram_id: int) -> str:
    """
    Get or create conversation ID for user.
    
    Args:
        telegram_id: Telegram user ID
        
    Returns:
        Conversation ID
    """
    if telegram_id not in user_conversations:
        user_conversations[telegram_id] = str(uuid.uuid4())
    return user_conversations[telegram_id]


def start_new_conversation(telegram_id: int) -> str:
    """
    Start new conversation for user.
    
    Args:
        telegram_id: Telegram user ID
        
    Returns:
        New conversation ID
    """
    user_conversations[telegram_id] = str(uuid.uuid4())
    return user_conversations[telegram_id]


async def call_rag_api(user_id: int, question: str, conversation_id: str = None, top_k: int = 3) -> dict:
    """
    Call RAG API to get answer.
    
    Args:
        user_id: User ID
        question: Question text
        conversation_id: Optional conversation ID for context
        top_k: Number of chunks to retrieve
        
    Returns:
        API response dict
    """
    try:
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            url = f"http://localhost:8000/api/v1/chat/query"
            payload = {
                "user_id": user_id,
                "question": question,
                "top_k": top_k,
                "conversation_id": conversation_id
            }
            
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"RAG API response: citations={len(result.get('citations', []))}, chunks={len(result.get('retrieved_chunks', []))}")
                    return result
                else:
                    logger.error(f"RAG API error: {response.status}")
                    return {"answer": "Sorry, I couldn't process your question right now."}
                    
    except Exception as e:
        logger.error(f"Failed to call RAG API: {e}")
        return {"answer": "Sorry, I couldn't process your question right now."}


@router.message(CommandStart())
async def start_command(message: Message):
    """Handle /start command."""
    try:
        user = await get_or_create_user(message.from_user.id)
        
        if user:
            # Start new conversation
            start_new_conversation(message.from_user.id)
            
            # Role-specific welcome message
            if user.role == 'admin':
                # Check if this is a super admin
                is_super_admin = str(message.from_user.id) in settings.telegram.admin_telegram_ids
                
                if is_super_admin:
                    welcome_text = f"""👋 <b>Добро пожаловать, {message.from_user.first_name}!</b>

Вы супер-администратор системы. У вас есть полный доступ ко всем функциям.

<b>Команды супер-администратора:</b>
• <code>/upload</code> - Загрузить документ (PDF, DOCX, TXT, MD)
• <code>/list</code> - Показать все документы
• <code>/delete &lt;doc_id&gt;</code> - Удалить документ
• <code>/ask &lt;вопрос&gt;</code> - Задать конкретный вопрос
• <code>/addadmin &lt;telegram_id&gt;</code> - Добавить администратора
• <code>/removeadmin &lt;telegram_id&gt;</code> - Удалить администратора
• <code>/listadmins</code> - Показать список администраторов

<b>Или просто напишите свой вопрос</b> - я отвечу на основе документов и запомню наш разговор!"""
                else:
                    welcome_text = f"""👋 <b>Добро пожаловать, {message.from_user.first_name}!</b>

Вы администратор системы. У вас есть доступ к основным функциям.

<b>Команды администратора:</b>
• <code>/upload</code> - Загрузить документ (PDF, DOCX, TXT, MD)
• <code>/list</code> - Показать все документы
• <code>/delete &lt;doc_id&gt;</code> - Удалить документ
• <code>/ask &lt;вопрос&gt;</code> - Задать конкретный вопрос

<b>Или просто напишите свой вопрос</b> - я отвечу на основе документов и запомню наш разговор!"""
            else:
                welcome_text = f"""👋 <b>Добро пожаловать, {message.from_user.first_name}!</b>

Вы можете задавать вопросы на основе загруженных документов.

<b>Доступные команды:</b>
• <code>/ask &lt;вопрос&gt;</code> - Задать конкретный вопрос

<b>Или просто напишите свой вопрос</b> - я отвечу на основе документов и запомню наш разговор!"""
            
            await message.reply(
                welcome_text,
                parse_mode="HTML",
                disable_web_page_preview=True
            )
        else:
            await message.reply(
                "⚠️ Что-то пошло не так, попробуйте позже.",
                disable_web_page_preview=True
            )
            
    except Exception as e:
        logger.error(f"Start command failed: {e}")
        await message.reply(
            "⚠️ Something went wrong, please try again later.",
            disable_web_page_preview=True
        )


@router.message(Command("upload"))
async def upload_command(message: Message):
    """Handle /upload command (admin only)."""
    try:
        # Check admin permissions
        if not await require_admin(message):
            return
            
        user = await get_or_create_user(message.from_user.id)
        
        if not user:
            await message.reply(
                "⚠️ Что-то пошло не так, попробуйте позже.",
                disable_web_page_preview=True
            )
            return
        
        # Start new conversation after upload
        start_new_conversation(message.from_user.id)
        
        if not message.document:
            await message.reply(
                "📎 Пожалуйста, отправьте файл документа (PDF, DOCX, TXT, MD)",
                disable_web_page_preview=True
            )
            return
        
        # Check file type
        file_name = message.document.file_name.lower()
        allowed_extensions = ['.pdf', '.docx', '.txt', '.md']
        
        if not any(file_name.endswith(ext) for ext in allowed_extensions):
            await message.reply(
                "❌ Неподдерживаемый тип файла. Пожалуйста, отправьте PDF, DOCX, TXT или MD файлы.",
                disable_web_page_preview=True
            )
            return
        
        # Download file
        file_info = await message.bot.get_file(message.document.file_id)
        file_content = await message.bot.download_file(file_info.file_path)
        
        if not file_content:
            await message.reply(
                "❌ Не удалось загрузить файл.",
                disable_web_page_preview=True
            )
            return
        
        # Index document
        doc_id, filename = index_document(
            file_content.read(),
            message.document.file_name,
            user.id
        )
        
        if doc_id:
            await message.reply(
                f"✅ Документ проиндексирован!\n📄 <b>{filename}</b>\n🆔 ID: <code>{doc_id}</code>\n\n💬 Теперь вы можете задавать вопросы об этом документе!",
                parse_mode="HTML",
                disable_web_page_preview=True
            )
        else:
            await message.reply(
                "❌ Не удалось проиндексировать документ.",
                disable_web_page_preview=True
            )
            
    except Exception as e:
        logger.error(f"Upload command failed: {e}")
        await message.reply(
            "⚠️ Something went wrong, please try again later.",
            disable_web_page_preview=True
        )


@router.message(Command("list"))
async def list_command(message: Message):
    """Handle /list command (admin only)."""
    try:
        # Check admin permissions
        if not await require_admin(message):
            return
            
        user = await get_or_create_user(message.from_user.id)
        
        if not user:
            await message.reply(
                "⚠️ Что-то пошло не так, попробуйте позже.",
                disable_web_page_preview=True
            )
            return
        
        documents = get_user_documents(user.id)
        
        if not documents:
            await message.reply(
                "📂 У вас нет загруженных документов.",
                disable_web_page_preview=True
            )
            return
        
        response_text = "📚 <b>Ваши документы:</b>\n\n"
        
        for doc in documents:
            size_mb = doc['size_bytes'] / (1024 * 1024)
            public_url = doc.get('public_url')
            
            if public_url:
                response_text += f"📄 <a href=\"{public_url}\">{doc['filename']}</a>\n"
            else:
                response_text += f"📄 {doc['filename']} — (url отсутствует)\n"
            
            response_text += f"🆔 ID: <code>{doc['id']}</code>\n"
            response_text += f"📅 {doc['created_at'][:10]}\n"
            response_text += f"💾 {size_mb:.1f} MB\n\n"
        
        # Split if too long
        if len(response_text) > 4000:
            response_text = response_text[:4000] + "..."
        
        await message.reply(response_text, parse_mode="HTML", disable_web_page_preview=True)
        
    except Exception as e:
        logger.error(f"List command failed: {e}")
        await message.reply(
            "⚠️ Something went wrong, please try again later.",
            disable_web_page_preview=True
        )


@router.message(Command("delete"))
async def delete_command(message: Message):
    """Handle /delete command (admin only)."""
    try:
        # Check admin permissions
        if not await require_admin(message):
            return
            
        user = await get_or_create_user(message.from_user.id)
        
        if not user:
            await message.reply(
                "⚠️ Что-то пошло не так, попробуйте позже.",
                disable_web_page_preview=True
            )
            return
        
        # Extract doc_id from command
        command_parts = message.text.split()
        if len(command_parts) < 2:
            await message.reply(
                "❌ Пожалуйста, укажите ID документа: <code>/delete &lt;doc_id&gt;</code>",
                parse_mode="HTML",
                disable_web_page_preview=True
            )
            return
        
        doc_id = command_parts[1]
        
        # Get document info before deletion for confirmation message
        from app.db.models import Document
        db = SessionLocal()
        try:
            doc = db.query(Document).filter(Document.id == doc_id).first()
            filename = doc.filename if doc else "Unknown"
            public_url = doc.public_url if doc else None
        finally:
            db.close()
        
        # Delete document
        success = delete_document(doc_id, user.id)
        
        if success:
            if public_url:
                await message.reply(f"🗑️ Документ {filename} удален", disable_web_page_preview=True)
            else:
                await message.reply(
                    f"🗑️ Документ {doc_id} удален",
                    disable_web_page_preview=True
                )
        else:
            await message.reply(
                "❌ Документ не найден или у вас нет прав на его удаление.",
                disable_web_page_preview=True
            )
            
    except Exception as e:
        logger.error(f"Delete command failed: {e}")
        await message.reply(
            "⚠️ Something went wrong, please try again later.",
            disable_web_page_preview=True
        )


@router.message(Command("ask"))
async def ask_command(message: Message):
    """Handle /ask command."""
    try:
        user = await get_or_create_user(message.from_user.id)
        
        if not user:
            await message.reply(
                "⚠️ Что-то пошло не так, попробуйте позже.",
                disable_web_page_preview=True
            )
            return
        
        # Extract question from command
        command_parts = message.text.split(maxsplit=1)
        if len(command_parts) < 2:
            await message.reply(
                "❌ Пожалуйста, укажите вопрос: <code>/ask &lt;вопрос&gt;</code>",
                parse_mode="HTML",
                disable_web_page_preview=True
            )
            return
        
        question = command_parts[1]
        
        # Get conversation ID
        conversation_id = get_user_conversation_id(message.from_user.id)
        
        # Show typing indicator
        await message.bot.send_chat_action(message.chat.id, "typing")
        
        # Call RAG API with conversation context
        response = await call_rag_api(user.id, question, conversation_id)
        
        # Format response
        answer = response.get("answer", "Извините, я не смог обработать ваш вопрос.")
        retrieved_chunks = response.get("retrieved_chunks", [])
        
        response_text = f"🤖 Answer:\n{answer}"
        
        if retrieved_chunks:
            response_text += "\n\n📚 Источники:\n"
            # Deduplicate sources by doc_id
            seen_docs = set()
            for chunk in retrieved_chunks[:3]:  # Limit to 3 sources
                doc_id = chunk['doc_id']
                if doc_id not in seen_docs:
                    seen_docs.add(doc_id)
                    # Use URL if available, otherwise show filename only
                    download_url = chunk.get('url') or chunk.get('public_url')
                    if download_url and not download_url.startswith('doc_id:'):
                        response_text += f"📄 {chunk['filename']} — {download_url}\n"
                    else:
                        response_text += f"📄 {chunk['filename']} — (url отсутствует)\n"
                    
                    # Add image links if available
                    if chunk.get('has_image') and chunk.get('image_urls'):
                        image_urls = chunk['image_urls']
                        if image_urls:
                            response_text += f"🖼️ Изображения: {', '.join(image_urls[:2])}\n"  # Limit to 2 images
        
        # Split if too long
        if len(response_text) > 4000:
            response_text = response_text[:4000] + "..."
        
        await message.reply(
            response_text,
            disable_web_page_preview=True
        )
        
    except Exception as e:
        logger.error(f"Ask command failed: {e}")
        await message.reply(
            "⚠️ Something went wrong, please try again later.",
            disable_web_page_preview=True
        )


@router.message(Command("addadmin"))
async def addadmin_command(message: Message):
    """Handle /addadmin command (super admin only)."""
    try:
        # Check super admin permissions
        if not await require_super_admin(message):
            return
        
        # Extract telegram_id from command
        command_parts = message.text.split()
        if len(command_parts) < 2:
            await message.reply(
                "❌ Пожалуйста, укажите Telegram ID пользователя: <code>/addadmin &lt;telegram_id&gt;</code>\n\n"
                "Пример: <code>/addadmin 123456789</code>",
                parse_mode="HTML",
                disable_web_page_preview=True
            )
            return
        
        try:
            telegram_id = int(command_parts[1])
        except ValueError:
            await message.reply(
                "❌ Неверный формат Telegram ID. Используйте только цифры.\n\n"
                "Пример: <code>/addadmin 123456789</code>",
                parse_mode="HTML",
                disable_web_page_preview=True
            )
            return
        
        # Check if trying to add self
        if telegram_id == message.from_user.id:
            await message.reply(
                "❌ Вы уже являетесь администратором.",
                disable_web_page_preview=True
            )
            return
        
        # Add admin
        success = await add_admin_to_db(telegram_id)
        
        if success:
            await message.reply(
                f"✅ Пользователь <code>{telegram_id}</code> успешно добавлен в администраторы!",
                parse_mode="HTML",
                disable_web_page_preview=True
            )
            logger.info(f"Super admin {message.from_user.id} added admin {telegram_id}")
        else:
            await message.reply(
                "❌ Не удалось добавить пользователя в администраторы. Попробуйте позже.",
                disable_web_page_preview=True
            )
        
    except Exception as e:
        logger.error(f"Addadmin command failed: {e}")
        await message.reply(
            "⚠️ Что-то пошло не так, попробуйте позже.",
            disable_web_page_preview=True
        )


@router.message(Command("removeadmin"))
async def removeadmin_command(message: Message):
    """Handle /removeadmin command (super admin only)."""
    try:
        # Check super admin permissions
        if not await require_super_admin(message):
            return
        
        # Extract telegram_id from command
        command_parts = message.text.split()
        if len(command_parts) < 2:
            await message.reply(
                "❌ Пожалуйста, укажите Telegram ID пользователя: <code>/removeadmin &lt;telegram_id&gt;</code>\n\n"
                "Пример: <code>/removeadmin 123456789</code>",
                parse_mode="HTML",
                disable_web_page_preview=True
            )
            return
        
        try:
            telegram_id = int(command_parts[1])
        except ValueError:
            await message.reply(
                "❌ Неверный формат Telegram ID. Используйте только цифры.\n\n"
                "Пример: <code>/removeadmin 123456789</code>",
                parse_mode="HTML",
                disable_web_page_preview=True
            )
            return
        
        # Check if trying to remove self
        if telegram_id == message.from_user.id:
            await message.reply(
                "❌ Вы не можете удалить себя из администраторов.",
                disable_web_page_preview=True
            )
            return
        
        # Remove admin
        success = await remove_admin_from_db(telegram_id)
        
        if success:
            await message.reply(
                f"✅ Пользователь <code>{telegram_id}</code> успешно удален из администраторов!",
                parse_mode="HTML",
                disable_web_page_preview=True
            )
            logger.info(f"Super admin {message.from_user.id} removed admin {telegram_id}")
        else:
            await message.reply(
                "❌ Не удалось удалить пользователя из администраторов. Возможно, пользователь не найден.",
                disable_web_page_preview=True
            )
        
    except Exception as e:
        logger.error(f"Removeadmin command failed: {e}")
        await message.reply(
            "⚠️ Что-то пошло не так, попробуйте позже.",
            disable_web_page_preview=True
        )


@router.message(Command("listadmins"))
async def listadmins_command(message: Message):
    """Handle /listadmins command (super admin only)."""
    try:
        # Check super admin permissions
        if not await require_super_admin(message):
            return
        
        admins = await get_all_admins()
        
        if not admins:
            await message.reply(
                "📋 Список администраторов пуст.",
                disable_web_page_preview=True
            )
            return
        
        response_text = "👥 <b>Список администраторов:</b>\n\n"
        
        for admin in admins:
            telegram_id = admin['telegram_id']
            created_at = admin['created_at']
            
            # Check if this is a super admin (from settings)
            is_super_admin = telegram_id in settings.telegram.admin_telegram_ids
            super_admin_badge = " 🔑" if is_super_admin else ""
            
            response_text += f"🆔 <code>{telegram_id}</code>{super_admin_badge}\n"
            if created_at:
                response_text += f"📅 {created_at[:10]}\n"
            response_text += "\n"
        
        # Add super admin info
        response_text += f"\n🔑 <b>Супер-администраторы</b> (из настроек):\n"
        for super_admin_id in settings.telegram.admin_telegram_ids:
            response_text += f"🆔 <code>{super_admin_id}</code>\n"
        
        await message.reply(
            response_text,
            parse_mode="HTML",
            disable_web_page_preview=True
        )
        
    except Exception as e:
        logger.error(f"Listadmins command failed: {e}")
        await message.reply(
            "⚠️ Что-то пошло не так, попробуйте позже.",
            disable_web_page_preview=True
        )


@router.message(F.document)
async def document_message(message: Message):
    """Handle document uploads."""
    try:
        user = await get_or_create_user(message.from_user.id)
        
        if not user:
            await message.reply(
                "⚠️ Что-то пошло не так, попробуйте позже.",
                disable_web_page_preview=True
            )
            return
        
        # Start new conversation after upload
        start_new_conversation(message.from_user.id)
        
        # Check file type
        file_name = message.document.file_name.lower()
        allowed_extensions = ['.pdf', '.docx', '.txt', '.md']
        
        if not any(file_name.endswith(ext) for ext in allowed_extensions):
            await message.reply(
                "❌ Неподдерживаемый тип файла. Пожалуйста, отправьте PDF, DOCX, TXT или MD файлы.",
                disable_web_page_preview=True
            )
            return
        
        # Download file
        file_info = await message.bot.get_file(message.document.file_id)
        file_content = await message.bot.download_file(file_info.file_path)
        
        if not file_content:
            await message.reply(
                "❌ Не удалось загрузить файл.",
                disable_web_page_preview=True
            )
            return
        
        # Index document
        doc_id, filename = index_document(
            file_content.read(),
            message.document.file_name,
            user.id
        )
        
        if doc_id:
            await message.reply(
                f"✅ Документ проиндексирован!\n📄 <b>{filename}</b>\n🆔 ID: <code>{doc_id}</code>\n\n💬 Теперь вы можете задавать вопросы об этом документе!",
                parse_mode="HTML",
                disable_web_page_preview=True
            )
        else:
            await message.reply(
                "❌ Не удалось проиндексировать документ.",
                disable_web_page_preview=True
            )
            
    except Exception as e:
        logger.error(f"Document upload failed: {e}")
        await message.reply(
                "⚠️ Что-то пошло не так, попробуйте позже.",
                disable_web_page_preview=True
            )


@router.message(F.voice)
async def voice_message(message: Message):
    """Handle voice messages."""
    try:
        await message.reply(
            "🎤 К сожалению, я пока не умею обрабатывать голосовые сообщения. Пожалуйста, отправьте ваш вопрос текстом.",
            disable_web_page_preview=True
        )
    except Exception as e:
        logger.error(f"Voice message handler failed: {e}")
        await message.reply(
                "⚠️ Что-то пошло не так, попробуйте позже.",
                disable_web_page_preview=True
            )


@router.message(F.sticker)
async def sticker_message(message: Message):
    """Handle sticker messages."""
    try:
        await message.reply(
            "😊 Спасибо за стикер! Но я лучше понимаю текстовые сообщения. Пожалуйста, напишите ваш вопрос.",
            disable_web_page_preview=True
        )
    except Exception as e:
        logger.error(f"Sticker message handler failed: {e}")
        await message.reply(
                "⚠️ Что-то пошло не так, попробуйте позже.",
                disable_web_page_preview=True
            )


@router.message(F.photo)
async def photo_message(message: Message):
    """Handle photo messages."""
    try:
        await message.reply(
            "📸 К сожалению, я пока не умею анализировать изображения. Пожалуйста, опишите ваш вопрос текстом или отправьте документ.",
            disable_web_page_preview=True
        )
    except Exception as e:
        logger.error(f"Photo message handler failed: {e}")
        await message.reply(
                "⚠️ Что-то пошло не так, попробуйте позже.",
                disable_web_page_preview=True
            )


@router.message(F.video)
async def video_message(message: Message):
    """Handle video messages."""
    try:
        await message.reply(
            "🎥 К сожалению, я пока не умею анализировать видео. Пожалуйста, опишите ваш вопрос текстом или отправьте документ.",
            disable_web_page_preview=True
        )
    except Exception as e:
        logger.error(f"Video message handler failed: {e}")
        await message.reply(
                "⚠️ Что-то пошло не так, попробуйте позже.",
                disable_web_page_preview=True
            )


@router.message(F.animation)
async def animation_message(message: Message):
    """Handle animation/GIF messages."""
    try:
        await message.reply(
            "🎬 К сожалению, я пока не умею анализировать анимации. Пожалуйста, опишите ваш вопрос текстом.",
            disable_web_page_preview=True
        )
    except Exception as e:
        logger.error(f"Animation message handler failed: {e}")
        await message.reply(
                "⚠️ Что-то пошло не так, попробуйте позже.",
                disable_web_page_preview=True
            )


@router.message(F.audio)
async def audio_message(message: Message):
    """Handle audio messages."""
    try:
        await message.reply(
            "🎵 К сожалению, я пока не умею обрабатывать аудио сообщения. Пожалуйста, отправьте ваш вопрос текстом.",
            disable_web_page_preview=True
        )
    except Exception as e:
        logger.error(f"Audio message handler failed: {e}")
        await message.reply(
                "⚠️ Что-то пошло не так, попробуйте позже.",
                disable_web_page_preview=True
            )


@router.message(F.location)
async def location_message(message: Message):
    """Handle location messages."""
    try:
        await message.reply(
            "📍 К сожалению, я пока не умею обрабатывать геолокацию. Пожалуйста, опишите ваш вопрос текстом.",
            disable_web_page_preview=True
        )
    except Exception as e:
        logger.error(f"Location message handler failed: {e}")
        await message.reply(
                "⚠️ Что-то пошло не так, попробуйте позже.",
                disable_web_page_preview=True
            )


@router.message(F.contact)
async def contact_message(message: Message):
    """Handle contact messages."""
    try:
        await message.reply(
            "📞 К сожалению, я пока не умею обрабатывать контакты. Пожалуйста, опишите ваш вопрос текстом.",
            disable_web_page_preview=True
        )
    except Exception as e:
        logger.error(f"Contact message handler failed: {e}")
        await message.reply(
                "⚠️ Что-то пошло не так, попробуйте позже.",
                disable_web_page_preview=True
            )


@router.message(F.text)
async def text_message(message: Message):
    """Handle text messages (free-form questions with conversation context)."""
    try:
        logger.info(f"Bot received text message: {message.text[:50]}... from user {message.from_user.id}")
        
        # Skip if message starts with command
        if message.text.startswith('/'):
            return
        
        user = await get_or_create_user(message.from_user.id)
        
        if not user:
            await message.reply(
                "⚠️ Что-то пошло не так, попробуйте позже.",
                disable_web_page_preview=True
            )
            return
        
        question = message.text.strip()
        
        if not question:
            return
        
        # Get conversation ID for context
        conversation_id = get_user_conversation_id(message.from_user.id)
        
        # Show typing indicator
        await message.bot.send_chat_action(message.chat.id, "typing")
        
        # Call RAG API with conversation context
        response = await call_rag_api(user.id, question, conversation_id)
        
        # Format response
        answer = response.get("answer", "Извините, я не смог обработать ваш вопрос.")
        retrieved_chunks = response.get("retrieved_chunks", [])
        citations = response.get("citations", [])
        
        logger.info(f"Bot received response: answer={answer[:100]}..., citations={len(citations)}, chunks={len(retrieved_chunks)}")
        
        response_text = f"🤖 {answer}"
        
        # Send main answer
        await message.reply(
            response_text,
            disable_web_page_preview=True
        )
        
        # Send citations with plain URLs if available
        logger.info(f"Citations received: {citations}")
        if citations:
            citations_text = "\n📚 Источники:\n"
            for citation in citations[:3]:  # Limit to 3 sources
                # Use new download URL if available, fallback to public_url
                download_url = citation.get('url') or citation.get('public_url')
                logger.info(f"Citation: {citation['filename']}, URL: {download_url}")
                if download_url and not download_url.startswith('doc_id:'):
                    citations_text += f"📄 {citation['filename']} — {download_url}\n"
                else:
                    citations_text += f"📄 {citation['filename']} — (url отсутствует)\n"
            
            logger.info(f"Sending citations: {citations_text}")
            await message.reply(citations_text, disable_web_page_preview=True)
        else:
            logger.warning("No citations received from API")
        
        # Send images from retrieved chunks
        if retrieved_chunks:
            await send_images_from_chunks(message, retrieved_chunks)
        
    except Exception as e:
        logger.error(f"Text message failed: {e}")
        await message.reply(
            "⚠️ Something went wrong, please try again later.",
            disable_web_page_preview=True
        )


@router.message()
async def unknown_message(message: Message):
    """Handle unknown message types."""
    try:
        await message.reply(
            "❓ К сожалению, я не понимаю этот тип сообщения. Пожалуйста, отправьте текстовое сообщение или документ.",
            disable_web_page_preview=True
        )
    except Exception as e:
        logger.error(f"Unknown message handler failed: {e}")
        await message.reply(
                "⚠️ Что-то пошло не так, попробуйте позже.",
                disable_web_page_preview=True
            )


def register_handlers(dp):
    """Register all handlers with dispatcher."""
    dp.include_router(router)
    logger.info("Telegram bot handlers registered")








