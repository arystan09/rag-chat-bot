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

# Store active conversations per user (fallback if DB session storage unavailable)
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
    """
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == str(telegram_id)).first()
        # Dynamic role assignment only if enabled
        desired_role = 'admin' if (settings.telegram.dynamic_roles and str(telegram_id) in settings.telegram.admin_telegram_ids) else 'user'
        if not user:
            user = User(telegram_id=str(telegram_id), role=desired_role)
            db.add(user)
            db.commit()
            db.refresh(user)
            logger.info(f"user_created", extra={"user_id": telegram_id, "role": desired_role})
        else:
            if settings.telegram.dynamic_roles and user.role != desired_role:
                old_role = user.role
                user.role = desired_role
                db.commit()
                logger.info(f"user_role_updated", extra={"user_id": telegram_id, "from": old_role, "to": desired_role})
        return user
    except Exception as e:
        logger.error(f"get_or_create_user_failed: {e}", extra={"user_id": telegram_id})
        db.rollback()
        return None
    finally:
        db.close()


def get_user_conversation_id(telegram_id: int) -> str:
    """Get or create conversation ID for user (in-memory fallback)."""
    if telegram_id not in user_conversations:
        user_conversations[telegram_id] = str(uuid.uuid4())
    return user_conversations[telegram_id]


def start_new_conversation(telegram_id: int) -> str:
    """Start new conversation (in-memory fallback)."""
    user_conversations[telegram_id] = str(uuid.uuid4())
    return user_conversations[telegram_id]


async def call_rag_api(user_id: int, question: str, conversation_id: str = None, top_k: int = 3) -> dict:
    """
    Call RAG API to get answer.
    Returns structured dict with error flag.
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
                    logger.info(
                        "rag_api_ok",
                        extra={
                            "user_id": user_id,
                            "conversation_id": conversation_id,
                            "citations": len(result.get('citations', [])),
                            "chunks": len(result.get('retrieved_chunks', []))
                        }
                    )
                    return {
                        "answer": result.get("answer"),
                        "citations": result.get("citations", []),
                        "retrieved_chunks": result.get("retrieved_chunks", []),
                        "error": False,
                        "raw": result
                    }
                else:
                    logger.error("rag_api_http_error", extra={"status": response.status})
                    return {"answer": "⚠️ Не удалось обработать запрос.", "citations": [], "retrieved_chunks": [], "error": True}
    except Exception as e:
        logger.error("rag_api_exception", extra={"error": str(e)})
        return {"answer": "⚠️ Не удалось обработать запрос.", "citations": [], "retrieved_chunks": [], "error": True}


async def process_document_upload(message: Message, user: User) -> None:
    """
    Common logic to process a document upload from a Message.
    Non-blocking indexing via asyncio.to_thread.
    """
    try:
        # Check file type
        file_name = message.document.file_name.lower()
        allowed_extensions = ['.pdf', '.docx', '.txt', '.md']
        if not any(file_name.endswith(ext) for ext in allowed_extensions):
            await message.reply(
                "❌ Неподдерживаемый тип файла. Пожалуйста, отправьте PDF, DOCX, TXT или MD файлы.",
                disable_web_page_preview=True
            )
            return
        
        file_info = await message.bot.get_file(message.document.file_id)
        file_content = await message.bot.download_file(file_info.file_path)
        if not file_content:
            await message.reply("❌ Не удалось загрузить файл.", disable_web_page_preview=True)
            return
        
        # Non-blocking indexing
        def _do_index():
            return index_document(file_content.read(), message.document.file_name, user.id)
        doc_id, filename = await asyncio.to_thread(_do_index)
        
        if doc_id:
            await message.reply(
                f"✅ Документ проиндексирован!\n📄 <b>{filename}</b>\n🆔 ID: <code>{doc_id}</code>\n\n💬 Теперь вы можете задавать вопросы об этом документе!",
                parse_mode="HTML",
                disable_web_page_preview=True
            )
        else:
            await message.reply("❌ Не удалось проиндексировать документ.", disable_web_page_preview=True)
    except Exception as e:
        logger.error("process_document_upload_failed", extra={"user_id": user.id, "error": str(e)})
        await message.reply("⚠️ Произошла ошибка при обработке файла. Попробуйте позже.")


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
        if not await require_admin(message):
            return
        user = await get_or_create_user(message.from_user.id)
        if not user:
            await message.reply("⚠️ Что-то пошло не так, попробуйте позже.", disable_web_page_preview=True)
            return
        start_new_conversation(message.from_user.id)
        if not message.document:
            await message.reply("📎 Пожалуйста, отправьте файл документа (PDF, DOCX, TXT, MD)", disable_web_page_preview=True)
            return
        await process_document_upload(message, user)
    except Exception as e:
        logger.error("upload_command_failed", extra={"user_id": message.from_user.id, "error": str(e)})
        await message.reply("⚠️ Произошла ошибка. Попробуйте позже.")


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
        if len(response_text) > settings.telegram.max_message_length:
            response_text = response_text[:settings.telegram.max_message_length] + "..."
        
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
            await message.reply("⚠️ Что-то пошло не так, попробуйте позже.", disable_web_page_preview=True)
            return
        command_parts = message.text.split(maxsplit=1)
        if len(command_parts) < 2:
            await message.reply("❌ Пожалуйста, укажите вопрос: <code>/ask &lt;вопрос&gt;</code>", parse_mode="HTML", disable_web_page_preview=True)
            return
        question = command_parts[1]
        conversation_id = get_user_conversation_id(message.from_user.id)
        await message.bot.send_chat_action(message.chat.id, "typing")
        resp = await call_rag_api(user.id, question, conversation_id)
        answer = resp.get("answer", "Извините, я не смог обработать ваш вопрос.")
        retrieved_chunks = resp.get("retrieved_chunks", [])
        citations = resp.get("citations", [])
        # Merge answer + single citation if fits
        text = f"🤖 {answer}"
        if citations:
            # Always show only the single best source
            c = citations[0]
            fn = c.get('filename') or 'Источник'
            url = c.get('url') or c.get('public_url')
            if not url or str(url).startswith('doc_id:'):
                cite_line = f"📄 {fn} — (url отсутствует)"
        else:
                cite_line = f"📄 {fn} — {url}"
            cite_block = "\n\n📚 Источники:\n" + cite_line
            merged = text + cite_block
            if len(merged) <= settings.telegram.max_message_length:
                text = merged
        if len(text) > settings.telegram.max_message_length:
            text = text[:settings.telegram.max_message_length] + "..."
        await message.reply(text, disable_web_page_preview=True)
        if retrieved_chunks:
            await send_images_from_chunks(message, retrieved_chunks)
    except Exception as e:
        logger.error("ask_command_failed", extra={"user_id": message.from_user.id, "error": str(e)})
        await message.reply("⚠️ Что-то пошло не так, попробуйте позже.", disable_web_page_preview=True)


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
            await message.reply("⚠️ Что-то пошло не так, попробуйте позже.", disable_web_page_preview=True)
            return
        start_new_conversation(message.from_user.id)
        await process_document_upload(message, user)
    except Exception as e:
        logger.error("document_message_failed", extra={"user_id": message.from_user.id, "error": str(e)})
        await message.reply("⚠️ Произошла ошибка при загрузке документа. Попробуйте позже.")


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
        logger.info("text_message", extra={"user_id": message.from_user.id, "text": message.text[:80]})
        if message.text.startswith('/'):
            return
        user = await get_or_create_user(message.from_user.id)
        if not user:
            await message.reply("⚠️ Что-то пошло не так, попробуйте позже.", disable_web_page_preview=True)
            return
        question = (message.text or '').strip()
        if not question:
            return
        conversation_id = get_user_conversation_id(message.from_user.id)
        await message.bot.send_chat_action(message.chat.id, "typing")
        resp = await call_rag_api(user.id, question, conversation_id)
        answer = resp.get("answer", "Извините, я не смог обработать ваш вопрос.")
        retrieved_chunks = resp.get("retrieved_chunks", [])
        citations = resp.get("citations", [])
        # Merge answer + single citation when fits
        text = f"🤖 {answer}"
        if citations:
            # Always show only the single best source
            c = citations[0]
            fn = c.get('filename') or 'Источник'
            url = c.get('url') or c.get('public_url')
            if not url or str(url).startswith('doc_id:'):
                cite_line = f"📄 {fn} — (url отсутствует)"
                else:
                cite_line = f"📄 {fn} — {url}"
            cite_block = "\n\n📚 Источники:\n" + cite_line
            merged = text + cite_block
            if len(merged) <= settings.telegram.max_message_length:
                text = merged
        if len(text) > settings.telegram.max_message_length:
            text = text[:settings.telegram.max_message_length] + "..."
        await message.reply(text, disable_web_page_preview=True)
        if retrieved_chunks:
            await send_images_from_chunks(message, retrieved_chunks)
    except Exception as e:
        logger.error("text_message_failed", extra={"user_id": message.from_user.id, "error": str(e)})
        await message.reply("⚠️ Что-то пошло не так, попробуйте позже.", disable_web_page_preview=True)


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








